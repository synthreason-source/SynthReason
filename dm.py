import os
import tkinter as tk
from tkinter import ttk, messagebox
import tweepy
from dotenv import load_dotenv
import requests
from threading import Thread
from datetime import datetime

load_dotenv()

# X/Twitter setup
bearer_token = os.getenv('BEARER_TOKEN')
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

# Config - EDIT THESE
TARGET_USERNAME = 'GeorgeW479744'  # Feed to monitor
FILTER_KEYWORD = 'AI'         # Posts must contain this
MAX_TWEETS = 10              # Recent tweets to check
RECIPIENTS = ['GeorgeW479744']  # DM targets (@ not needed)

# Globals
relevant_tweets = []
recipient_ids = {}  # Cache: username -> id

def fetch_tweets():
    """Extract and prefilter tweets from target user."""
    global relevant_tweets
    try:
        user = client.get_user(username=TARGET_USERNAME)
        if not user.data:
            return False, "Target user not found"
        
        tweets_resp = client.get_users_tweets(
            id=user.data.id,
            max_results=MAX_TWEETS,
            tweet_fields=['created_at', 'public_metrics']
        )
        
        relevant_tweets = []
        if tweets_resp.data:
            for tweet in tweets_resp.data:
                if FILTER_KEYWORD.lower() in tweet.text.lower():
                    relevant_tweets.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at.isoformat()
                    })
        return True, f"Found {len(relevant_tweets)} relevant tweets from @{TARGET_USERNAME}"
    except Exception as e:
        return False, f"Fetch error: {e}"

def get_user_id(username):
    """Get/cached user ID."""
    if username in recipient_ids:
        return recipient_ids[username]
    try:
        user = client.get_user(username=username)
        if user.data:
            recipient_ids[username] = user.data.id
            return user.data.id
    except:
        pass
    return None

def generate_dm(username, tweets_context):
    """Generate personalized DM text via Perplexity API (direct HTTP)."""
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY missing from .env")
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""Generate a short, friendly DM (<280 chars) responding to @{TARGET_USERNAME}'s recent {FILTER_KEYWORD} posts:
{ tweets_context }

Personalize for @{username}: show genuine interest, reference their interests if possible, ask ONE engaging question.
Natural tone, no spam vibes."""
    
    data = {
        "model": "sonar-medium-online",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200
    }
    
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    content = resp.json()['choices'][0]['message']['content'].strip()
    return content[:277] + "..." if len(content) > 277 else content  # Safety trim

def send_dms_thread(selected_usernames):
    """Send DMs in background thread."""
    if not relevant_tweets:
        messagebox.showerror("No Tweets", "Fetch tweets first!")
        return
    
    success_count = 0
    tweets_context = "\n".join([t['text'] for t in relevant_tweets[:3]])  # Top 3
    
    for username in selected_usernames:
        print(f"Processing @{username}...")
        
        user_id = get_user_id(username)
        if not user_id:
            print(f"❌ Skipping @{username}: User not found")
            continue
        
        try:
            dm_text = generate_dm(username, tweets_context)
            print(f"📝 Generated: {dm_text[:100]}...")
            
            dm = client.create_direct_message(
                participant_ids=[str(user_id)],
                text=dm_text
            )
            print(f"✅ DM sent to @{username}! ID: {dm.data.id}")
            success_count += 1
        except tweepy.Forbidden:
            print(f"❌ DM failed to @{username}: No permission (must follow you?)")
        except requests.RequestException as e:
            print(f"❌ Perplexity error for @{username}: {e}")
        except Exception as e:
            print(f"❌ Error to @{username}: {e}")
    
    messagebox.showinfo("DMs Sent", f"✅ {success_count}/{len(selected_usernames)} successful!")

def open_dialog():
    """Main Tkinter dialog."""
    fetch_ok, msg = fetch_tweets()
    if not fetch_ok:
        messagebox.showerror("Error", msg)
        return
    
    dialog = tk.Toplevel()
    dialog.title("DM Generator - Select Recipients")
    dialog.geometry("450x450")
    dialog.resizable(False, False)
    
    # Header
    header_frame = ttk.Frame(dialog)
    header_frame.pack(pady=10, fill=tk.X, padx=20)
    ttk.Label(header_frame, text="🚀 X DM Automator", font=('Arial', 14, 'bold')).pack()
    ttk.Label(header_frame, text=f"{msg} | {datetime.now().strftime('%H:%M:%S')}", 
              font=('Arial', 9)).pack()
    
    # Listbox for multi-select
    list_frame = ttk.Frame(dialog)
    list_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
    ttk.Label(list_frame, text="Hold Ctrl/Cmd to select multiple:", font=('Arial', 10)).pack(anchor=tk.W)
    
    listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=12, font=('Arial', 10))
    for username in RECIPIENTS:
        listbox.insert(tk.END, f"@{username}")
    listbox.pack(fill=tk.BOTH, expand=True)
    
    # Buttons
    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(pady=20)
    
    def send_action():
        selected_idx = listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("Selection", "Select at least one recipient!")
            return
        selected = [RECIPIENTS[i] for i in selected_idx]
        dialog.destroy()
        Thread(target=send_dms_thread, args=(selected,), daemon=True).start()
    
    ttk.Button(btn_frame, text="📤 Send Selected DMs", command=send_action).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_frame, text="🔄 Refresh Tweets", command=lambda: open_dialog()).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_frame, text="❌ Close", command=dialog.destroy).pack(side=tk.LEFT)
    
    dialog.transient(tk.Tk())
    dialog.grab_set()
    dialog.wait_window()

# Run
if __name__ == "__main__":
    if not all([bearer_token, consumer_key, access_token, os.getenv('PERPLEXITY_API_KEY')]):
        print("❌ Missing keys in .env! Add: BEARER_TOKEN, CONSUMER_KEY, etc.")
    else:
        root = tk.Tk()
        root.withdraw()
        open_dialog()
        root.destroy()
