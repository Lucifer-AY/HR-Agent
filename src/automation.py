from googleapiclient.discovery import build
from google.oauth2 import service_account
from twilio.rest import Client
from slack_sdk import WebClient
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(filename='data/logs/automation.log', level=logging.INFO)

# Google Calendar setup
creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=['https://www.googleapis.com/auth/calendar'])
calendar_service = build('calendar', 'v3', credentials=creds)

# Twilio and Slack setup
twilio_client = Client("twilio-account-sid", "twilio-auth-token")
slack_client = WebClient(token="slack-bot-token")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def schedule_interview(candidate_email, hr_email, interview_time):
    """Schedule an interview with retry mechanism."""
    event = {
        'summary': 'Candidate Interview',
        'description': 'Automated interview scheduled by HR Tech Suite',
        'start': {'dateTime': interview_time, 'timeZone': 'UTC'},
        'end': {'dateTime': f"{interview_time.split('T')[0]}T{int(interview_time.split('T')[1][:2]) + 1:02d}:00:00Z", 'timeZone': 'UTC'},
        'attendees': [{'email': candidate_email}, {'email': hr_email}],
        'reminders': {'useDefault': False, 'overrides': [{'method': 'email', 'minutes': 30}]}
    }
    try:
        event_result = calendar_service.events().insert(calendarId='primary', body=event).execute()
        logging.info(f"Scheduled interview: {event_result['id']}")
        return event_result['id']
    except Exception as e:
        logging.error(f"Scheduling failed: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_sms(phone_number, message):
    """Send SMS with retry mechanism."""
    try:
        msg = twilio_client.messages.create(body=message, from_="+1234567890", to=phone_number)
        logging.info(f"SMS sent to {phone_number}: {msg.sid}")
    except Exception as e:
        logging.error(f"SMS failed: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def notify_hr_slack(message):
    """Notify HR via Slack with retry mechanism."""
    try:
        response = slack_client.chat_postMessage(channel="#hr-channel", text=message)
        logging.info(f"Slack notification sent: {response['ts']}")
    except Exception as e:
        logging.error(f"Slack notification failed: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    interview_time = "2023-12-01T10:00:00Z"
    event_id = schedule_interview("candidate@example.com", "hr@example.com", interview_time)
    send_sms("+1234567890", f"Your interview is scheduled for Dec 1st at 10 AM (Event ID: {event_id}).")
    notify_hr_slack(f"New candidate interview scheduled: {event_id}")