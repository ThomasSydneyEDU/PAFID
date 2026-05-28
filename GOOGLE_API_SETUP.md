# How to Get a Google Gemini API Key

To use the PAFID pipeline for food image generation, you need a Google Gemini API key. Google offers a generous free tier for researchers and developers through **Google AI Studio**.

## Step-by-Step Instructions

### 1. Sign in to Google AI Studio
1.  Go to [Google AI Studio (aistudio.google.com)](https://aistudio.google.com/).
2.  Sign in with your Google Account.

### 2. Create Your API Key
1.  On the left-hand sidebar, click on the **"Get API key"** button (indicated by a key icon).
2.  Click the blue button labeled **"Create API key in new project"**. 
    *   *Note: If you already have a Google Cloud project, you can select "Create API key in existing project" instead.*
3.  A dialog will appear with your new API key. **Copy this key immediately.** 
    *   *Warning: Treat this key like a password. Do not share it or commit it to a public GitHub repository.*

### 3. Set Up Your Environment
To use the key with the PAFID scripts, you need to set it as an environment variable on your computer.

#### On Mac or Linux (Terminal):
Add this line to your `~/.zshrc` or `~/.bash_profile` to make it permanent, or run it in your current session:
```bash
export GEMINI_API_KEY="your_actual_key_here"
```

#### On Windows (Command Prompt):
```cmd
setx GEMINI_API_KEY "your_actual_key_here"
```

#### On Windows (PowerShell):
```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your_actual_key_here", "User")
```

## Troubleshooting & Limits
*   **Free Tier**: As of mid-2024, Google AI Studio provides a free tier that allows for image generation (Imagen 3) with certain rate limits (RPM/RPD). 
*   **Region Availability**: Ensure the Gemini API is available in your country. If you encounter a "Region not supported" error, you may need to check Google's official documentation for the latest availability list.
*   **Safety Filters**: Gemini has built-in safety filters. If the pipeline reports that an image was "blocked," it is likely due to the model's internal safety guidelines regarding the prompt.

For more detailed documentation, visit the [Google Gemini API Documentation](https://ai.google.dev/docs).
