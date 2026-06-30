# Setting Up Google Gemini for PAFID

The PAFID pipeline supports two ways to authenticate with the Gemini API. **Vertex AI is the recommended option** for institutional and research use — it uses your Google Cloud project credentials and avoids API-key management. The AI Studio API key option is still supported for quick personal use.

---

## Option 1 — Vertex AI (Recommended)

Vertex AI uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) tied to your Google Cloud project. No API key is needed.

### Prerequisites
- A Google Cloud project with the Vertex AI API enabled
- The [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed

### Step 1 — Authenticate

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

### Step 2 — Set environment variables

Add these to your `~/.zshrc` or `~/.bash_profile` to make them permanent, or export them in your current session:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
```

On Windows (PowerShell):
```powershell
[System.Environment]::SetEnvironmentVariable("GOOGLE_CLOUD_PROJECT", "your-project-id", "User")
[System.Environment]::SetEnvironmentVariable("GOOGLE_CLOUD_LOCATION", "global", "User")
[System.Environment]::SetEnvironmentVariable("GOOGLE_GENAI_USE_VERTEXAI", "True", "User")
```

### Step 3 — Run the pipeline

```bash
python src/generate_stimuli.py --classify-only
```

At startup you should see:
```
[INFO] Using Gemini via Vertex AI: project=your-project-id, location=global
```

### Troubleshooting
- **"GOOGLE_CLOUD_PROJECT is not set"** — ensure you have exported the variable in the current shell session.
- **Permission errors** — make sure your Google Cloud account has the `Vertex AI User` role on the project, and that the Vertex AI API is enabled in the [Google Cloud Console](https://console.cloud.google.com/).
- **Quota / billing** — Vertex AI usage is billed to your Google Cloud project. Check quotas in the console under **Vertex AI > Quotas**.

---

## Option 2 — Gemini Developer API (AI Studio API Key)

This option uses a personal API key from Google AI Studio. Suitable for quick local use, but subject to per-project rate limits and requires managing a secret key.

### Step 1 — Create an API key

1. Go to [Google AI Studio](https://aistudio.google.com/) and sign in.
2. Click **"Get API key"** in the left sidebar, then **"Create API key"**.
3. Copy the key immediately. Treat it like a password — do not commit it to a public repository.

### Step 2 — Set the environment variable

On Mac or Linux:
```bash
export GEMINI_API_KEY="your_actual_key_here"
```

To make it permanent, add the line above to your `~/.zshrc` or `~/.bash_profile`.

On Windows (Command Prompt):
```cmd
setx GEMINI_API_KEY "your_actual_key_here"
```

On Windows (PowerShell):
```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your_actual_key_here", "User")
```

### Step 3 — Run the pipeline

Do **not** set `GOOGLE_GENAI_USE_VERTEXAI`, or set it to `False`:

```bash
python3 src/classify_food.py --limit 5
```

At startup you should see:
```
[INFO] Using Gemini Developer API with GEMINI_API_KEY
```

### Troubleshooting
- **"GEMINI_API_KEY is not set"** — ensure the variable is exported in your current shell session.
- **Rate limits** — the free tier has requests-per-minute and requests-per-day limits. If you hit them, wait and retry, or consider switching to Vertex AI.
- **Region availability** — if you see a "Region not supported" error, check [Google's availability list](https://ai.google.dev/docs).
- **Safety filters** — if an image is blocked, it is due to the model's built-in safety guidelines. Try rephrasing the prompt.

---

## Which option should I use?

| | Vertex AI | AI Studio API Key |
|---|---|---|
| Recommended for | Institutional / research use | Quick personal use |
| Authentication | Google Cloud ADC | API key string |
| API key management | Not required | Required |
| Billing | Google Cloud project | AI Studio quota |
| Rate limits | Configurable via Cloud quotas | Fixed free-tier limits |
