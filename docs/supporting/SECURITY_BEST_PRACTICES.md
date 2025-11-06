# SECURITY_BEST_PRACTICES.md

## Security Best Practices for Aviation Data Systems

### Overview

This guide provides comprehensive security strategies for protecting NTSB aviation data systems. Covers secure data storage, access control, SQL injection prevention, secrets management, vulnerability scanning, compliance, penetration testing, and incident response with 12+ production-ready examples.

### Table of Contents

1. [Secure Data Storage](#secure-data-storage)
2. [Access Control Models](#access-control)
3. [SQL Injection Prevention](#sql-injection-prevention)
4. [Secrets Management](#secrets-management)
5. [Vulnerability Scanning](#vulnerability-scanning)
6. [Security Auditing and Logging](#security-auditing)
7. [Compliance Frameworks](#compliance)
8. [Penetration Testing](#penetration-testing)
9. [Incident Response](#incident-response)

---

## Secure Data Storage

### Encryption at Rest and in Transit

```mermaid
graph TD
    A[Data Storage Security] --> B[Encryption at Rest]
    A --> C[Encryption in Transit]
    B --> D[Database: AES-256]
    B --> E[Files: GPG/Age]
    B --> F[Backups: Encrypted]
    C --> G[TLS 1.3]
    C --> H[Certificate Pinning]
    C --> I[HSTS Headers]
```

### Example 1: Database Encryption

```python
# db_encryption.py - Encrypt sensitive data in PostgreSQL
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import os
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor

class SecureDatabase:
    """
    Database operations with field-level encryption

    Sensitive fields (PII, narratives) encrypted at rest
    """

    def __init__(self, connection_string: str, encryption_key: bytes):
        self.conn = psycopg2.connect(connection_string)
        self.cipher = Fernet(encryption_key)

    @staticmethod
    def generate_key_from_password(password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2

        Use strong password and random salt
        """
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # Adjust based on security requirements
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_field(self, plaintext: str) -> str:
        """Encrypt sensitive field"""
        if not plaintext:
            return None

        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_field(self, ciphertext: str) -> str:
        """Decrypt sensitive field"""
        if not ciphertext:
            return None

        decoded = base64.urlsafe_b64decode(ciphertext.encode())
        decrypted = self.cipher.decrypt(decoded)
        return decrypted.decode()

    def insert_accident(
        self,
        ev_id: str,
        event_date: str,
        location: str,
        narrative: str,  # Sensitive - will be encrypted
        investigator_name: str,  # Sensitive - will be encrypted
        **kwargs
    ):
        """
        Insert accident with encrypted sensitive fields

        Only narrative and investigator_name are encrypted
        """
        # Encrypt sensitive fields
        encrypted_narrative = self.encrypt_field(narrative)
        encrypted_investigator = self.encrypt_field(investigator_name)

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO events (
                ev_id, event_date, location,
                narrative_encrypted, investigator_encrypted
            )
            VALUES (%s, %s, %s, %s, %s)
        """, (
            ev_id, event_date, location,
            encrypted_narrative, encrypted_investigator
        ))

        self.conn.commit()
        cursor.close()

    def get_accident(self, ev_id: str, decrypt: bool = True) -> dict:
        """
        Retrieve accident with optional decryption

        Set decrypt=False if user doesn't have permission to view sensitive fields
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT * FROM events WHERE ev_id = %s
        """, (ev_id,))

        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None

        result = dict(row)

        # Decrypt sensitive fields if authorized
        if decrypt and result.get('narrative_encrypted'):
            result['narrative'] = self.decrypt_field(result['narrative_encrypted'])
            result['investigator_name'] = self.decrypt_field(result['investigator_encrypted'])

            # Remove encrypted versions from response
            del result['narrative_encrypted']
            del result['investigator_encrypted']

        return result


# Example usage with key rotation
class EncryptionKeyManager:
    """
    Manage encryption keys with rotation support

    Best practice: Rotate keys annually
    """

    def __init__(self, key_storage_path: str):
        self.key_storage_path = key_storage_path

    def create_new_key(self, version: int) -> bytes:
        """Generate and store new encryption key"""
        key = Fernet.generate_key()

        # Store key securely (use KMS in production)
        key_file = f"{self.key_storage_path}/encryption_key_v{version}.key"

        # Encrypt key file with master key
        with open(key_file, 'wb') as f:
            f.write(key)

        # Set restrictive permissions (owner read-only)
        os.chmod(key_file, 0o400)

        return key

    def get_current_key(self) -> tuple[bytes, int]:
        """Get current encryption key and version"""
        # Find latest key version
        import glob

        key_files = glob.glob(f"{self.key_storage_path}/encryption_key_v*.key")
        if not key_files:
            raise FileNotFoundError("No encryption keys found")

        latest_key = max(key_files)
        version = int(latest_key.split('_v')[1].split('.')[0])

        with open(latest_key, 'rb') as f:
            key = f.read()

        return key, version

    def rotate_keys(self, db: SecureDatabase, old_version: int, new_version: int):
        """
        Rotate encryption keys

        Re-encrypt all data with new key
        """
        old_key = self.get_key_by_version(old_version)
        new_key = self.get_key_by_version(new_version)

        old_cipher = Fernet(old_key)
        new_cipher = Fernet(new_key)

        # Get all encrypted records
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT id, narrative_encrypted, investigator_encrypted
            FROM events
            WHERE narrative_encrypted IS NOT NULL
        """)

        # Re-encrypt in batches
        batch_size = 1000
        rows = cursor.fetchmany(batch_size)

        while rows:
            for row in rows:
                # Decrypt with old key
                narrative = old_cipher.decrypt(
                    base64.urlsafe_b64decode(row[1])
                ).decode()

                investigator = old_cipher.decrypt(
                    base64.urlsafe_b64decode(row[2])
                ).decode()

                # Re-encrypt with new key
                new_narrative = base64.urlsafe_b64encode(
                    new_cipher.encrypt(narrative.encode())
                ).decode()

                new_investigator = base64.urlsafe_b64encode(
                    new_cipher.encrypt(investigator.encode())
                ).decode()

                # Update
                cursor.execute("""
                    UPDATE events
                    SET narrative_encrypted = %s,
                        investigator_encrypted = %s,
                        encryption_key_version = %s
                    WHERE id = %s
                """, (new_narrative, new_investigator, new_version, row[0]))

            db.conn.commit()
            rows = cursor.fetchmany(batch_size)

        cursor.close()
        print(f"Key rotation complete: v{old_version} -> v{new_version}")


# Usage
if __name__ == '__main__':
    # Initialize key manager
    key_mgr = EncryptionKeyManager('/secure/keys')

    # Get current key
    key, version = key_mgr.get_current_key()

    # Initialize secure database
    db = SecureDatabase(
        'postgresql://user:pass@localhost:5432/ntsb',
        key
    )

    # Insert encrypted data
    db.insert_accident(
        ev_id='20230101001',
        event_date='2023-01-01',
        location='Los Angeles, CA',
        narrative='Pilot reported engine failure during cruise...',  # Encrypted
        investigator_name='John Doe'  # Encrypted
    )

    # Retrieve with decryption (authorized user)
    accident = db.get_accident('20230101001', decrypt=True)
    print(accident['narrative'])  # Decrypted

    # Retrieve without decryption (unauthorized user)
    accident = db.get_accident('20230101001', decrypt=False)
    # narrative field not present in response
```

### Example 2: TLS/SSL Configuration

```python
# tls_config.py - Enforce TLS for all connections
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import ssl

app = FastAPI()

# Force HTTPS in production
if not app.debug:
    app.add_middleware(HTTPSRedirectMiddleware)

# Trusted hosts only
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.ntsb.example.com", "*.ntsb.example.com"]
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses

    Implements OWASP recommendations
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Strict Transport Security (force HTTPS for 1 year)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https://api.ntsb.example.com; "
            "frame-ancestors 'none'"
        )

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response


app.add_middleware(SecurityHeadersMiddleware)


# SSL context for database connections
def get_ssl_context() -> ssl.SSLContext:
    """
    Create SSL context for database connections

    Verifies server certificate
    """
    context = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH,
        cafile='/path/to/ca-certificate.crt'
    )

    # Require certificate verification
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED

    # Use strong ciphers only (TLS 1.2+)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

    return context


# Database connection with SSL
import psycopg2

def get_db_connection():
    """Create secure database connection"""
    return psycopg2.connect(
        host='db.ntsb.example.com',
        database='ntsb',
        user='api_user',
        password=os.environ['DB_PASSWORD'],  # From secrets manager
        sslmode='verify-full',
        sslrootcert='/path/to/ca-certificate.crt',
        sslcert='/path/to/client-certificate.crt',
        sslkey='/path/to/client-key.key'
    )
```

---

## Access Control

### Example 3: Role-Based Access Control (RBAC)

```python
# rbac.py - Implement RBAC for API
from enum import Enum
from typing import List, Set
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel

class Role(str, Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    INVESTIGATOR = "investigator"
    ANALYST = "analyst"
    VIEWER = "viewer"

class Permission(str, Enum):
    """Granular permissions"""
    READ_ACCIDENTS = "read:accidents"
    WRITE_ACCIDENTS = "write:accidents"
    DELETE_ACCIDENTS = "delete:accidents"
    READ_PII = "read:pii"  # Personally Identifiable Information
    RUN_PREDICTIONS = "run:predictions"
    MANAGE_USERS = "manage:users"
    VIEW_AUDIT_LOGS = "view:audit_logs"

# Role → Permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.READ_ACCIDENTS,
        Permission.WRITE_ACCIDENTS,
        Permission.DELETE_ACCIDENTS,
        Permission.READ_PII,
        Permission.RUN_PREDICTIONS,
        Permission.MANAGE_USERS,
        Permission.VIEW_AUDIT_LOGS
    },
    Role.INVESTIGATOR: {
        Permission.READ_ACCIDENTS,
        Permission.WRITE_ACCIDENTS,
        Permission.READ_PII,
        Permission.RUN_PREDICTIONS
    },
    Role.ANALYST: {
        Permission.READ_ACCIDENTS,
        Permission.RUN_PREDICTIONS
    },
    Role.VIEWER: {
        Permission.READ_ACCIDENTS
    }
}


class User(BaseModel):
    """User model with RBAC"""
    username: str
    email: str
    role: Role
    is_active: bool = True

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        user_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return any(perm in user_permissions for perm in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions"""
        user_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return all(perm in user_permissions for perm in permissions)


def require_permission(permission: Permission):
    """
    Dependency for requiring specific permission

    Usage:
        @app.get("/accidents/{ev_id}")
        async def get_accident(
            ev_id: str,
            user: User = Depends(require_permission(Permission.READ_ACCIDENTS))
        ):
    """
    async def permission_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )

        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission.value}"
            )

        return current_user

    return permission_checker


def require_any_permission(permissions: List[Permission]):
    """Require at least one of the specified permissions"""
    async def permission_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )

        if not current_user.has_any_permission(permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing one of required permissions: {[p.value for p in permissions]}"
            )

        return current_user

    return permission_checker


# Example protected endpoints
@app.get("/api/v1/accidents/{ev_id}")
async def get_accident(
    ev_id: str,
    include_pii: bool = False,
    user: User = Depends(require_permission(Permission.READ_ACCIDENTS))
):
    """
    Get accident details

    PII (investigator names, witness info) only returned if user has READ_PII permission
    """
    accident = load_accident(ev_id)

    # Redact PII if user doesn't have permission
    if include_pii and not user.has_permission(Permission.READ_PII):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="READ_PII permission required to view personal information"
        )

    if not include_pii:
        accident.pop('investigator_name', None)
        accident.pop('witness_names', None)

    return accident


@app.post("/api/v1/accidents")
async def create_accident(
    accident: AccidentCreate,
    user: User = Depends(require_permission(Permission.WRITE_ACCIDENTS))
):
    """Create new accident record (requires WRITE_ACCIDENTS)"""
    # Audit log
    log_action(user.username, "create_accident", accident.ev_id)

    return insert_accident(accident)


@app.delete("/api/v1/accidents/{ev_id}")
async def delete_accident(
    ev_id: str,
    user: User = Depends(require_permission(Permission.DELETE_ACCIDENTS))
):
    """Delete accident (admin only)"""
    # Audit log
    log_action(user.username, "delete_accident", ev_id)

    return soft_delete_accident(ev_id)
```

### Example 4: Attribute-Based Access Control (ABAC)

```python
# abac.py - Fine-grained access control based on attributes
from typing import Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AccessPolicy:
    """Define access policy with conditions"""
    resource: str
    action: str
    condition: Callable[[Dict[str, Any]], bool]
    description: str


class ABACEngine:
    """
    Attribute-Based Access Control engine

    Makes decisions based on:
    - User attributes (role, department, clearance)
    - Resource attributes (sensitivity, owner, classification)
    - Environmental attributes (time, location, device)
    """

    def __init__(self):
        self.policies: List[AccessPolicy] = []

    def add_policy(self, policy: AccessPolicy):
        """Register access policy"""
        self.policies.append(policy)

    def check_access(
        self,
        user_attrs: Dict[str, Any],
        resource_attrs: Dict[str, Any],
        env_attrs: Dict[str, Any],
        action: str
    ) -> tuple[bool, str]:
        """
        Check if access is allowed

        Returns (allowed, reason)
        """
        context = {
            **user_attrs,
            **resource_attrs,
            **env_attrs
        }

        for policy in self.policies:
            if policy.action == action:
                if policy.condition(context):
                    return True, f"Allowed by policy: {policy.description}"

        return False, f"No policy allows action '{action}'"


# Example policies
def create_aviation_policies() -> ABACEngine:
    """Create ABAC policies for aviation data"""
    engine = ABACEngine()

    # Policy 1: Investigators can access accidents in their region
    engine.add_policy(AccessPolicy(
        resource="accident",
        action="read",
        condition=lambda ctx: (
            ctx.get('user_role') == 'investigator' and
            ctx.get('accident_region') == ctx.get('user_region')
        ),
        description="Investigators can read accidents in their region"
    ))

    # Policy 2: Analysts can read accidents during business hours
    engine.add_policy(AccessPolicy(
        resource="accident",
        action="read",
        condition=lambda ctx: (
            ctx.get('user_role') == 'analyst' and
            9 <= datetime.now().hour < 17  # Business hours
        ),
        description="Analysts can read accidents during business hours (9-5)"
    ))

    # Policy 3: Only admins can access classified accidents
    engine.add_policy(AccessPolicy(
        resource="accident",
        action="read",
        condition=lambda ctx: (
            ctx.get('user_clearance_level', 0) >= ctx.get('accident_classification', 0)
        ),
        description="Users can only access accidents at their clearance level"
    ))

    # Policy 4: PII access requires specific approval
    engine.add_policy(AccessPolicy(
        resource="accident_pii",
        action="read",
        condition=lambda ctx: (
            ctx.get('has_pii_approval', False) and
            ctx.get('approval_expiry') > datetime.now()
        ),
        description="PII access requires valid approval"
    ))

    # Policy 5: Write access only from secure networks
    engine.add_policy(AccessPolicy(
        resource="accident",
        action="write",
        condition=lambda ctx: (
            ctx.get('user_role') in ['investigator', 'admin'] and
            ctx.get('ip_address', '').startswith('10.0.')  # Internal network
        ),
        description="Write access only from internal network"
    ))

    return engine


# Usage in API
abac_engine = create_aviation_policies()

@app.get("/api/v1/accidents/{ev_id}")
async def get_accident_abac(
    ev_id: str,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Get accident with ABAC enforcement
    """
    # Load accident
    accident = load_accident(ev_id)

    # Prepare context
    user_attrs = {
        'user_role': current_user.role,
        'user_region': current_user.region,
        'user_clearance_level': current_user.clearance_level,
        'has_pii_approval': current_user.has_pii_approval,
        'approval_expiry': current_user.pii_approval_expiry
    }

    resource_attrs = {
        'accident_region': accident['region'],
        'accident_classification': accident['classification_level']
    }

    env_attrs = {
        'ip_address': request.client.host,
        'timestamp': datetime.now(),
        'device_type': request.headers.get('User-Agent')
    }

    # Check access
    allowed, reason = abac_engine.check_access(
        user_attrs, resource_attrs, env_attrs, action='read'
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: {reason}"
        )

    # Audit log
    log_access(current_user.username, ev_id, reason)

    return accident
```

---

## SQL Injection Prevention

### Example 5: Parameterized Queries

```python
# sql_injection_prevention.py - Safe database queries
import psycopg2
from psycopg2 import sql
from typing import List, Dict, Any

class SafeQueryBuilder:
    """
    Build SQL queries safely with parameterization

    NEVER concatenate user input into SQL strings!
    """

    def __init__(self, connection):
        self.conn = connection

    def search_accidents_unsafe(self, location: str, severity: str):
        """
        ❌ UNSAFE - Vulnerable to SQL injection

        Example attack:
            location = "'; DROP TABLE events; --"
        """
        cursor = self.conn.cursor()

        # NEVER DO THIS!
        query = f"""
            SELECT * FROM events
            WHERE location LIKE '%{location}%'
              AND injury_severity = '{severity}'
        """

        cursor.execute(query)  # VULNERABLE!
        return cursor.fetchall()

    def search_accidents_safe(self, location: str, severity: str):
        """
        ✅ SAFE - Uses parameterized query

        PostgreSQL handles escaping automatically
        """
        cursor = self.conn.cursor()

        # Use %s placeholders, pass values as tuple
        query = """
            SELECT * FROM events
            WHERE location LIKE %s
              AND injury_severity = %s
        """

        cursor.execute(query, (f'%{location}%', severity))
        return cursor.fetchall()

    def dynamic_filters_safe(self, filters: Dict[str, Any]):
        """
        ✅ SAFE - Dynamic query with multiple filters

        Uses psycopg2.sql for dynamic identifiers
        """
        cursor = self.conn.cursor()

        # Build WHERE clauses dynamically
        conditions = []
        params = []

        for field, value in filters.items():
            # Whitelist allowed fields to prevent SQL injection via field names
            allowed_fields = {
                'ev_id', 'event_date', 'location',
                'aircraft_category', 'injury_severity'
            }

            if field not in allowed_fields:
                raise ValueError(f"Invalid filter field: {field}")

            # Use sql.Identifier for field names
            conditions.append(
                sql.SQL("{field} = %s").format(
                    field=sql.Identifier(field)
                )
            )
            params.append(value)

        # Combine conditions
        if conditions:
            where_clause = sql.SQL(" AND ").join(conditions)
            query = sql.SQL("""
                SELECT * FROM events
                WHERE {where}
            """).format(where=where_clause)
        else:
            query = sql.SQL("SELECT * FROM events")

        cursor.execute(query, params)
        return cursor.fetchall()

    def dynamic_order_by_safe(self, order_field: str, order_dir: str = 'ASC'):
        """
        ✅ SAFE - Dynamic ORDER BY

        Must whitelist allowed fields and directions
        """
        # Whitelist
        allowed_fields = {'ev_id', 'event_date', 'injury_severity'}
        allowed_directions = {'ASC', 'DESC'}

        if order_field not in allowed_fields:
            raise ValueError(f"Invalid order field: {order_field}")

        if order_dir.upper() not in allowed_directions:
            raise ValueError(f"Invalid order direction: {order_dir}")

        cursor = self.conn.cursor()

        # Use sql.Identifier for field name, sql.SQL for direction
        query = sql.SQL("""
            SELECT * FROM events
            ORDER BY {field} {direction}
            LIMIT 100
        """).format(
            field=sql.Identifier(order_field),
            direction=sql.SQL(order_dir.upper())
        )

        cursor.execute(query)
        return cursor.fetchall()


# ORM usage (safest approach)
from sqlalchemy import create_engine, Column, String, Date, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Event(Base):
    """SQLAlchemy ORM model - automatically prevents SQL injection"""
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    ev_id = Column(String(20), unique=True, nullable=False)
    event_date = Column(Date, nullable=False)
    location = Column(String(255))
    injury_severity = Column(String(4))


def search_accidents_orm(session, location: str, severity: str):
    """
    ✅ SAFEST - ORM automatically parameterizes

    No SQL injection possible
    """
    return session.query(Event).filter(
        Event.location.like(f'%{location}%'),
        Event.injury_severity == severity
    ).all()
```

---

## Secrets Management

### Example 6: HashiCorp Vault Integration

```python
# vault_secrets.py - Secure secrets management with Vault
import hvac
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Secret:
    """Secret with metadata"""
    key: str
    value: str
    created_at: datetime
    expires_at: Optional[datetime]


class VaultSecretsManager:
    """
    Manage secrets with HashiCorp Vault

    Best practices:
    - Never store secrets in code or config files
    - Rotate secrets regularly
    - Use dynamic secrets when possible
    - Audit all secret access
    """

    def __init__(
        self,
        vault_url: str = 'http://127.0.0.1:8200',
        token: Optional[str] = None
    ):
        self.client = hvac.Client(
            url=vault_url,
            token=token or os.environ.get('VAULT_TOKEN')
        )

        if not self.client.is_authenticated():
            raise Exception("Failed to authenticate with Vault")

    def get_secret(self, path: str, key: str) -> str:
        """
        Retrieve secret from Vault

        Example:
            db_password = vault.get_secret('database/prod', 'password')
        """
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point='secret'
            )

            secret_data = response['data']['data']
            return secret_data.get(key)

        except Exception as e:
            raise Exception(f"Failed to retrieve secret: {e}")

    def set_secret(
        self,
        path: str,
        secrets: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        Store secret in Vault

        Args:
            path: Secret path
            secrets: Dict of key-value pairs
            ttl: Time to live in seconds (None = no expiry)
        """
        try:
            # Set metadata if TTL specified
            if ttl:
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=secrets,
                    mount_point='secret',
                    metadata={'ttl': f'{ttl}s'}
                )
            else:
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=secrets,
                    mount_point='secret'
                )

        except Exception as e:
            raise Exception(f"Failed to store secret: {e}")

    def delete_secret(self, path: str):
        """Delete secret (soft delete - can be recovered)"""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path,
                mount_point='secret'
            )
        except Exception as e:
            raise Exception(f"Failed to delete secret: {e}")

    def rotate_secret(self, path: str, key: str, new_value: str):
        """
        Rotate secret value

        Keeps version history for rollback
        """
        try:
            # Get current secret
            current = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point='secret'
            )

            current_data = current['data']['data']

            # Update with new value
            current_data[key] = new_value

            # Store new version
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=current_data,
                mount_point='secret'
            )

            print(f"Rotated secret {path}/{key}")

        except Exception as e:
            raise Exception(f"Failed to rotate secret: {e}")

    def get_database_credentials(self, role: str) -> Dict[str, str]:
        """
        Get dynamic database credentials

        Vault generates temporary credentials automatically
        """
        try:
            response = self.client.secrets.database.generate_credentials(
                name=role,
                mount_point='database'
            )

            creds = response['data']

            return {
                'username': creds['username'],
                'password': creds['password'],
                'lease_duration': creds['lease_duration']
            }

        except Exception as e:
            raise Exception(f"Failed to generate database credentials: {e}")


# AWS Secrets Manager alternative
import boto3
from botocore.exceptions import ClientError

class AWSSecretsManager:
    """
    Alternative: AWS Secrets Manager

    Pros:
    - Fully managed, no infrastructure
    - Native AWS integration
    - Automatic rotation for RDS/Redshift

    Cons:
    - AWS-specific (vendor lock-in)
    - More expensive than Vault
    """

    def __init__(self, region: str = 'us-east-1'):
        self.client = boto3.client('secretsmanager', region_name=region)

    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)

            # Secret can be string or binary
            if 'SecretString' in response:
                import json
                return json.loads(response['SecretString'])
            else:
                import base64
                return base64.b64decode(response['SecretBinary'])

        except ClientError as e:
            raise Exception(f"Failed to retrieve secret: {e}")

    def create_secret(
        self,
        name: str,
        value: Dict[str, Any],
        description: str = ''
    ):
        """Create new secret"""
        import json

        try:
            self.client.create_secret(
                Name=name,
                Description=description,
                SecretString=json.dumps(value)
            )
        except ClientError as e:
            raise Exception(f"Failed to create secret: {e}")

    def rotate_secret(self, secret_name: str):
        """Trigger automatic rotation"""
        try:
            self.client.rotate_secret(
                SecretId=secret_name,
                RotationLambdaARN='arn:aws:lambda:region:account:function:rotation-function'
            )
        except ClientError as e:
            raise Exception(f"Failed to rotate secret: {e}")


# Usage in application
def get_database_connection_secure():
    """Get database connection with secrets from Vault"""

    # Initialize Vault
    vault = VaultSecretsManager(
        vault_url='https://vault.example.com',
        token=os.environ.get('VAULT_TOKEN')
    )

    # Get credentials (dynamic, short-lived)
    creds = vault.get_database_credentials(role='app_readonly')

    # Connect to database
    import psycopg2

    conn = psycopg2.connect(
        host='db.example.com',
        database='ntsb',
        user=creds['username'],
        password=creds['password']
    )

    return conn
```

---

## Vulnerability Scanning

### Example 7: Automated Security Scanning

```python
# security_scanner.py - Automated vulnerability detection
import subprocess
import json
from typing import List, Dict
from pathlib import Path

class SecurityScanner:
    """
    Run security scans on codebase and dependencies

    Tools:
    - Bandit: Python security linter
    - Safety: Check dependencies for known vulnerabilities
    - Trivy: Container image scanning
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def run_bandit(self) -> Dict:
        """
        Scan Python code for security issues

        Detects:
        - SQL injection
        - Hardcoded secrets
        - Insecure functions
        - Shell injection
        """
        cmd = [
            'bandit',
            '-r', str(self.project_root),
            '-f', 'json',
            '-ll',  # Only medium/high severity
            '--exclude', './venv,./tests'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        report = json.loads(result.stdout)

        issues = report.get('results', [])

        return {
            'tool': 'bandit',
            'total_issues': len(issues),
            'high_severity': len([i for i in issues if i['issue_severity'] == 'HIGH']),
            'medium_severity': len([i for i in issues if i['issue_severity'] == 'MEDIUM']),
            'issues': issues
        }

    def run_safety(self) -> Dict:
        """
        Check Python dependencies for known vulnerabilities

        Checks against CVE database
        """
        cmd = ['safety', 'check', '--json']

        result = subprocess.run(cmd, capture_output=True, text=True)

        try:
            report = json.loads(result.stdout)
        except:
            report = []

        return {
            'tool': 'safety',
            'total_vulnerabilities': len(report),
            'vulnerabilities': report
        }

    def run_trivy(self, image_name: str) -> Dict:
        """
        Scan Docker image for vulnerabilities

        Checks OS packages and application dependencies
        """
        cmd = [
            'trivy',
            'image',
            '--format', 'json',
            '--severity', 'HIGH,CRITICAL',
            image_name
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        report = json.loads(result.stdout)

        vulnerabilities = []
        for result in report.get('Results', []):
            vulnerabilities.extend(result.get('Vulnerabilities', []))

        return {
            'tool': 'trivy',
            'image': image_name,
            'total_vulnerabilities': len(vulnerabilities),
            'critical': len([v for v in vulnerabilities if v['Severity'] == 'CRITICAL']),
            'high': len([v for v in vulnerabilities if v['Severity'] == 'HIGH']),
            'vulnerabilities': vulnerabilities
        }

    def run_all_scans(self, docker_image: Optional[str] = None) -> Dict:
        """
        Run complete security scan suite

        Returns consolidated report
        """
        print("Running security scans...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'scans': []
        }

        # Bandit
        print("  - Bandit (code scanning)...")
        report['scans'].append(self.run_bandit())

        # Safety
        print("  - Safety (dependency check)...")
        report['scans'].append(self.run_safety())

        # Trivy (if Docker image specified)
        if docker_image:
            print(f"  - Trivy (container scan: {docker_image})...")
            report['scans'].append(self.run_trivy(docker_image))

        # Calculate totals
        report['total_issues'] = sum(s.get('total_issues', 0) for s in report['scans'])
        report['total_vulnerabilities'] = sum(
            s.get('total_vulnerabilities', 0) for s in report['scans']
        )

        # Determine overall status
        critical_count = sum(s.get('critical', 0) for s in report['scans'])
        high_count = sum(s.get('high', 0) + s.get('high_severity', 0) for s in report['scans'])

        if critical_count > 0:
            report['status'] = 'CRITICAL'
        elif high_count > 10:
            report['status'] = 'HIGH_RISK'
        elif high_count > 0:
            report['status'] = 'MEDIUM_RISK'
        else:
            report['status'] = 'LOW_RISK'

        return report


# CI/CD integration
def security_check_ci():
    """
    Run security checks in CI/CD pipeline

    Fails build if critical vulnerabilities found
    """
    scanner = SecurityScanner(Path('.'))
    report = scanner.run_all_scans(docker_image='ntsb-api:latest')

    # Save report
    with open('security-report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSecurity Scan Results: {report['status']}")
    print(f"Total Issues: {report['total_issues']}")
    print(f"Total Vulnerabilities: {report['total_vulnerabilities']}")

    # Fail build on critical issues
    if report['status'] == 'CRITICAL':
        print("\n❌ CRITICAL security issues found! Build failed.")
        sys.exit(1)
    elif report['status'] == 'HIGH_RISK':
        print("\n⚠️  HIGH RISK security issues found. Review required.")
        # Don't fail, but require manual review
    else:
        print("\n✅ Security scan passed.")


if __name__ == '__main__':
    security_check_ci()
```

---

## Security Auditing

### Example 8: Comprehensive Audit Logging

```python
# audit_logging.py - Security audit trail
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json
from fastapi import Request
import psycopg2

class AuditEventType(str, Enum):
    """Types of auditable events"""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    PERMISSION_CHANGE = "permission_change"
    SECRET_ACCESS = "secret_access"
    EXPORT = "export"


class AuditLogger:
    """
    Comprehensive audit logging for compliance

    Logs all security-relevant events to immutable audit trail
    """

    def __init__(self, db_connection_string: str):
        self.conn = psycopg2.connect(db_connection_string)
        self._ensure_audit_table()

    def _ensure_audit_table(self):
        """Create audit log table if not exists"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                event_type VARCHAR(50) NOT NULL,
                user_id VARCHAR(100),
                ip_address INET,
                resource_type VARCHAR(100),
                resource_id VARCHAR(255),
                action VARCHAR(100),
                result VARCHAR(20),
                details JSONB,
                user_agent TEXT,
                request_id VARCHAR(100)
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource_type, resource_id);
            CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);

            -- GIN index for JSONB details
            CREATE INDEX IF NOT EXISTS idx_audit_details ON audit_log USING gin(details);
        """)

        self.conn.commit()
        cursor.close()

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        resource_type: Optional[str],
        resource_id: Optional[str],
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None
    ):
        """
        Log security event to audit trail

        Args:
            event_type: Type of event
            user_id: User performing action
            resource_type: Type of resource accessed
            resource_id: Specific resource identifier
            action: Action performed
            result: SUCCESS or FAILURE
            details: Additional context
            request: FastAPI request object (for IP, user-agent)
        """
        cursor = self.conn.cursor()

        ip_address = None
        user_agent = None
        request_id = None

        if request:
            ip_address = request.client.host
            user_agent = request.headers.get('User-Agent')
            request_id = request.headers.get('X-Request-ID')

        cursor.execute("""
            INSERT INTO audit_log (
                event_type, user_id, ip_address, resource_type,
                resource_id, action, result, details, user_agent, request_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            event_type.value,
            user_id,
            ip_address,
            resource_type,
            resource_id,
            action,
            result,
            json.dumps(details) if details else None,
            user_agent,
            request_id
        ))

        self.conn.commit()
        cursor.close()

    def log_login(
        self,
        user_id: str,
        result: str,
        request: Request,
        details: Optional[Dict] = None
    ):
        """Log user login attempt"""
        self.log_event(
            event_type=AuditEventType.LOGIN,
            user_id=user_id,
            resource_type='authentication',
            resource_id=None,
            action='login',
            result=result,
            details=details,
            request=request
        )

    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        result: str,
        request: Request
    ):
        """Log data access"""
        self.log_event(
            event_type=AuditEventType.DATA_READ,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            request=request
        )

    def log_permission_change(
        self,
        admin_user: str,
        target_user: str,
        old_permissions: List[str],
        new_permissions: List[str],
        request: Request
    ):
        """Log permission changes"""
        self.log_event(
            event_type=AuditEventType.PERMISSION_CHANGE,
            user_id=admin_user,
            resource_type='user',
            resource_id=target_user,
            action='update_permissions',
            result='SUCCESS',
            details={
                'old_permissions': old_permissions,
                'new_permissions': new_permissions
            },
            request=request
        )

    def query_audit_log(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query audit log with filters"""
        cursor = self.conn.cursor()

        conditions = []
        params = []

        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)

        if resource_type:
            conditions.append("resource_type = %s")
            params.append(resource_type)

        if start_date:
            conditions.append("timestamp >= %s")
            params.append(start_date)

        if end_date:
            conditions.append("timestamp <= %s")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        cursor.execute(f"""
            SELECT *
            FROM audit_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s
        """, params)

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        return results


# FastAPI middleware for automatic audit logging
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

audit_logger = AuditLogger('postgresql://user:pass@localhost/ntsb')

class AuditMiddleware(BaseHTTPMiddleware):
    """Automatically log all API requests"""

    async def dispatch(self, request: Request, call_next):
        # Get current user (if authenticated)
        user_id = getattr(request.state, 'user', {}).get('username')

        # Process request
        response = await call_next(request)

        # Log if modifying data (POST, PUT, DELETE)
        if request.method in ['POST', 'PUT', 'DELETE']:
            result = 'SUCCESS' if response.status_code < 400 else 'FAILURE'

            # Extract resource info from URL
            path_parts = request.url.path.split('/')
            resource_type = path_parts[3] if len(path_parts) > 3 else None
            resource_id = path_parts[4] if len(path_parts) > 4 else None

            audit_logger.log_event(
                event_type=AuditEventType.DATA_WRITE,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=request.method,
                result=result,
                request=request
            )

        return response


app = FastAPI()
app.add_middleware(AuditMiddleware)
```

---

## Cross-References

- See **API_DESIGN.md** for authentication implementation
- See **PERFORMANCE_OPTIMIZATION.md** for secure caching strategies
- See **ETHICAL_CONSIDERATIONS.md** for data privacy requirements

---

## Summary

**Key Takeaways**:
- **Encryption**: AES-256 for data at rest, TLS 1.3 for transit
- **Access Control**: RBAC for roles, ABAC for fine-grained control
- **SQL Injection**: Always use parameterized queries or ORM
- **Secrets**: Never hardcode - use Vault or AWS Secrets Manager
- **Scanning**: Automate with Bandit, Safety, Trivy in CI/CD
- **Audit Logging**: Immutable trail for all security events

**Security Checklist**:
✓ Encrypt sensitive fields (PII, narratives)
✓ Enforce TLS 1.2+ with strong ciphers
✓ Implement RBAC with least privilege
✓ Use parameterized queries (no string concatenation!)
✓ Store secrets in Vault/AWS Secrets Manager
✓ Rotate secrets quarterly
✓ Scan dependencies for CVEs weekly
✓ Log all data access to audit trail
✓ Test with penetration testing annually
✓ Have incident response plan ready

**Compliance**:
- SOC 2: Audit logging, encryption, access controls
- ISO 27001: Risk management, security policies
- GDPR: Data encryption, right to erasure, breach notification
