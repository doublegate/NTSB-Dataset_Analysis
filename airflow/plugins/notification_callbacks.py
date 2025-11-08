"""
Notification callbacks for NTSB Aviation Data Sync DAG.

Provides Slack and email alerting for DAG failures, warnings, and success.
Compatible with Apache Airflow 2.x callback system.

Author: NTSB ETL Team
Created: 2025-11-07
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import requests
from airflow.models import TaskInstance
from airflow.utils.email import send_email


# =============================================================================
# CONFIGURATION
# =============================================================================

# Slack webhook URLs (set via environment variables or Airflow connections)
SLACK_WEBHOOK_CRITICAL = os.getenv("SLACK_WEBHOOK_CRITICAL", "")
SLACK_WEBHOOK_INFO = os.getenv("SLACK_WEBHOOK_INFO", "")

# Email configuration (uses Airflow SMTP settings from airflow.cfg or .env)
EMAIL_RECIPIENTS_CRITICAL = os.getenv("EMAIL_RECIPIENTS_CRITICAL", "").split(",")
EMAIL_RECIPIENTS_INFO = os.getenv("EMAIL_RECIPIENTS_INFO", "").split(",")

# Airflow Web UI base URL
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")


# =============================================================================
# SLACK NOTIFICATION FUNCTIONS
# =============================================================================

def send_slack_message(
    webhook_url: str,
    title: str,
    message: str,
    severity: str = "INFO",
    color: str = "#36a64f",
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send a formatted message to Slack using webhook.

    Args:
        webhook_url: Slack incoming webhook URL
        title: Message title/header
        message: Main message body
        severity: Alert severity (CRITICAL, WARNING, INFO)
        color: Slack attachment color (hex code)
        context: Optional Airflow context dictionary

    Returns:
        True if message sent successfully, False otherwise
    """
    if not webhook_url:
        print(f"⚠️  No Slack webhook configured, skipping notification: {title}")
        return False

    # Build context fields if available
    fields = []
    if context:
        task_instance: TaskInstance = context.get("task_instance")
        if task_instance:
            fields.extend([
                {
                    "title": "DAG",
                    "value": task_instance.dag_id,
                    "short": True,
                },
                {
                    "title": "Task",
                    "value": task_instance.task_id,
                    "short": True,
                },
                {
                    "title": "Execution Date",
                    "value": str(task_instance.execution_date),
                    "short": True,
                },
                {
                    "title": "Try Number",
                    "value": str(task_instance.try_number),
                    "short": True,
                },
            ])

            # Add log URL
            log_url = task_instance.log_url
            if log_url:
                fields.append({
                    "title": "Logs",
                    "value": f"<{AIRFLOW_BASE_URL}{log_url}|View Logs>",
                    "short": False,
                })

    # Build Slack payload
    payload = {
        "attachments": [
            {
                "color": color,
                "title": f"[{severity}] {title}",
                "text": message,
                "fields": fields,
                "footer": "NTSB Aviation Data Sync",
                "ts": int(datetime.now().timestamp()),
            }
        ]
    }

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        print(f"✅ Slack notification sent: {title}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Slack notification: {e}")
        return False


def send_slack_alert_critical(context: Dict[str, Any]) -> None:
    """
    Send CRITICAL alert to Slack for DAG/task failures.

    This callback is triggered automatically when a task fails.

    Args:
        context: Airflow context dictionary (automatically provided)
    """
    task_instance: TaskInstance = context.get("task_instance")
    exception = context.get("exception")

    title = f"DAG Failure: {task_instance.dag_id}"
    message = (
        f"❌ **Task `{task_instance.task_id}` failed**\n\n"
        f"**Error**: {str(exception)[:500] if exception else 'Unknown error'}\n\n"
        f"**Action Required**: Investigate logs and resolve issue before next scheduled run."
    )

    send_slack_message(
        webhook_url=SLACK_WEBHOOK_CRITICAL,
        title=title,
        message=message,
        severity="CRITICAL",
        color="#ff0000",  # Red
        context=context,
    )


def send_slack_alert_warning(context: Dict[str, Any], warning_message: str) -> None:
    """
    Send WARNING alert to Slack for non-critical issues.

    Args:
        context: Airflow context dictionary
        warning_message: Custom warning message
    """
    task_instance: TaskInstance = context.get("task_instance")

    title = f"DAG Warning: {task_instance.dag_id}"
    message = (
        f"⚠️  **Warning in task `{task_instance.task_id}`**\n\n"
        f"{warning_message}\n\n"
        f"**Action**: Review and address if needed."
    )

    send_slack_message(
        webhook_url=SLACK_WEBHOOK_CRITICAL,  # Warnings go to same channel
        title=title,
        message=message,
        severity="WARNING",
        color="#ffa500",  # Orange
        context=context,
    )


def send_slack_success(context: Dict[str, Any]) -> None:
    """
    Send SUCCESS notification to Slack for completed DAG runs.

    This should be called from the final task using PythonOperator.

    Args:
        context: Airflow context dictionary (automatically provided)
    """
    task_instance: TaskInstance = context.get("task_instance")
    dag_run = context.get("dag_run")

    # Calculate metrics from XCom if available
    events_loaded = task_instance.xcom_pull(task_ids="load_new_data", key="events_loaded") or "N/A"
    duplicates_found = task_instance.xcom_pull(task_ids="load_new_data", key="duplicates_found") or "N/A"

    # Calculate duration
    if dag_run and dag_run.start_date and dag_run.end_date:
        duration_seconds = (dag_run.end_date - dag_run.start_date).total_seconds()
        duration_str = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
    else:
        duration_str = "N/A"

    title = f"DAG Success: {task_instance.dag_id}"
    message = (
        f"✅ **Monthly NTSB data sync completed successfully!**\n\n"
        f"**Metrics**:\n"
        f"• Events Loaded: {events_loaded}\n"
        f"• Duplicates Found: {duplicates_found}\n"
        f"• Duration: {duration_str}\n"
        f"• Execution Date: {task_instance.execution_date}\n\n"
        f"**Next Run**: Scheduled for 1st of next month at 2 AM"
    )

    send_slack_message(
        webhook_url=SLACK_WEBHOOK_INFO,
        title=title,
        message=message,
        severity="INFO",
        color="#36a64f",  # Green
        context=context,
    )


# =============================================================================
# EMAIL NOTIFICATION FUNCTIONS
# =============================================================================

def send_email_alert_critical(context: Dict[str, Any]) -> None:
    """
    Send CRITICAL alert via email for DAG/task failures.

    Uses Airflow's built-in email configuration (SMTP settings from .env).

    Args:
        context: Airflow context dictionary (automatically provided)
    """
    task_instance: TaskInstance = context.get("task_instance")
    exception = context.get("exception")

    if not EMAIL_RECIPIENTS_CRITICAL or not EMAIL_RECIPIENTS_CRITICAL[0]:
        print("⚠️  No email recipients configured, skipping email notification")
        return

    subject = f"[CRITICAL] NTSB DAG Failure: {task_instance.task_id}"

    # Build HTML email body
    log_url = f"{AIRFLOW_BASE_URL}{task_instance.log_url}" if task_instance.log_url else "N/A"

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #ff0000; color: white; padding: 10px; }}
            .content {{ padding: 20px; }}
            .footer {{ background-color: #f0f0f0; padding: 10px; font-size: 12px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>❌ CRITICAL: NTSB Data Sync DAG Failed</h2>
        </div>
        <div class="content">
            <p><strong>A critical failure occurred in the NTSB Aviation Data Sync pipeline.</strong></p>

            <table>
                <tr><th>DAG</th><td>{task_instance.dag_id}</td></tr>
                <tr><th>Task</th><td>{task_instance.task_id}</td></tr>
                <tr><th>Execution Date</th><td>{task_instance.execution_date}</td></tr>
                <tr><th>Try Number</th><td>{task_instance.try_number}</td></tr>
                <tr><th>Error</th><td>{str(exception)[:500] if exception else 'Unknown error'}</td></tr>
                <tr><th>Logs</th><td><a href="{log_url}">View Logs</a></td></tr>
            </table>

            <p><strong>Action Required:</strong></p>
            <ul>
                <li>Review logs for detailed error information</li>
                <li>Resolve underlying issue</li>
                <li>Re-run DAG manually if needed</li>
                <li>Ensure next scheduled run (1st of month) succeeds</li>
            </ul>
        </div>
        <div class="footer">
            NTSB Aviation Data Sync Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """

    try:
        send_email(
            to=EMAIL_RECIPIENTS_CRITICAL,
            subject=subject,
            html_content=html_content,
        )
        print(f"✅ Email alert sent to {len(EMAIL_RECIPIENTS_CRITICAL)} recipient(s)")
    except Exception as e:
        print(f"❌ Failed to send email alert: {e}")


def send_email_success(context: Dict[str, Any]) -> None:
    """
    Send SUCCESS notification via email for completed DAG runs.

    Args:
        context: Airflow context dictionary (automatically provided)
    """
    task_instance: TaskInstance = context.get("task_instance")
    dag_run = context.get("dag_run")

    if not EMAIL_RECIPIENTS_INFO or not EMAIL_RECIPIENTS_INFO[0]:
        print("⚠️  No email recipients configured for success notifications")
        return

    # Calculate metrics
    events_loaded = task_instance.xcom_pull(task_ids="load_new_data", key="events_loaded") or "N/A"
    duplicates_found = task_instance.xcom_pull(task_ids="load_new_data", key="duplicates_found") or "N/A"

    if dag_run and dag_run.start_date and dag_run.end_date:
        duration_seconds = (dag_run.end_date - dag_run.start_date).total_seconds()
        duration_str = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"
    else:
        duration_str = "N/A"

    subject = f"[SUCCESS] NTSB DAG Completed: {task_instance.dag_id}"

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #36a64f; color: white; padding: 10px; }}
            .content {{ padding: 20px; }}
            .footer {{ background-color: #f0f0f0; padding: 10px; font-size: 12px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>✅ SUCCESS: NTSB Data Sync Completed</h2>
        </div>
        <div class="content">
            <p><strong>Monthly NTSB Aviation data sync completed successfully!</strong></p>

            <table>
                <tr><th>DAG</th><td>{task_instance.dag_id}</td></tr>
                <tr><th>Execution Date</th><td>{task_instance.execution_date}</td></tr>
                <tr><th>Duration</th><td>{duration_str}</td></tr>
                <tr><th>Events Loaded</th><td>{events_loaded}</td></tr>
                <tr><th>Duplicates Found</th><td>{duplicates_found}</td></tr>
            </table>

            <p><strong>Next Run:</strong> Scheduled for 1st of next month at 2:00 AM</p>
        </div>
        <div class="footer">
            NTSB Aviation Data Sync Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """

    try:
        send_email(
            to=EMAIL_RECIPIENTS_INFO,
            subject=subject,
            html_content=html_content,
        )
        print(f"✅ Success email sent to {len(EMAIL_RECIPIENTS_INFO)} recipient(s)")
    except Exception as e:
        print(f"❌ Failed to send success email: {e}")


# =============================================================================
# COMBINED CALLBACK FUNCTIONS (Slack + Email)
# =============================================================================

def notify_failure(context: Dict[str, Any]) -> None:
    """
    Combined failure notification (Slack + Email).

    Use this as on_failure_callback in DAG default_args.
    """
    send_slack_alert_critical(context)
    send_email_alert_critical(context)


def notify_success(context: Dict[str, Any]) -> None:
    """
    Combined success notification (Slack + Email).

    Use this in the final task of the DAG.
    """
    send_slack_success(context)
    # Email success notifications are optional (comment out if too noisy)
    # send_email_success(context)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_slack_webhook(webhook_url: str) -> bool:
    """
    Test a Slack webhook with a simple message.

    Args:
        webhook_url: Slack incoming webhook URL

    Returns:
        True if test successful, False otherwise
    """
    return send_slack_message(
        webhook_url=webhook_url,
        title="Test Notification",
        message="✅ Slack webhook is configured correctly!",
        severity="INFO",
        color="#36a64f",
    )


if __name__ == "__main__":
    """
    Test script - run with:
    python notification_callbacks.py
    """
    print("Testing Slack webhook configuration...")

    webhook = os.getenv("SLACK_WEBHOOK_CRITICAL")
    if webhook:
        success = test_slack_webhook(webhook)
        print(f"Test result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    else:
        print("⚠️  SLACK_WEBHOOK_CRITICAL not set in environment")
        print("Set webhook URL and try again:")
        print("export SLACK_WEBHOOK_CRITICAL='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'")
