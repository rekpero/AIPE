# AIPE: AI Pipeline Engine

## 🚀 Introduction

AIPE (AI Pipeline Engine) is a flexible and powerful tool for creating and executing complex AI workflows. It allows you to chain together various AI tasks such as web searching, text generation, speech processing, and image generation into a cohesive pipeline, all configurable through a simple YAML file.

## ✨ Features

- 🔗 Chain multiple AI tasks into a single workflow
- 🔍 Web search integration (DuckDuckGo and Serper API)
- 🤖 Text generation using various LLM models (Ollama, OpenAI)
- 🗣️ Text-to-Speech and Speech-to-Text capabilities
- 🖼️ Image generation using Stable Diffusion
- 🌐 Webhook integration for external API calls
- 📄 YAML-based configuration for easy pipeline setup
- 🔄 Dynamic binding of context and step outputs

## 🚀 Use Cases

AIPE (AI Pipeline Engine) is a versatile tool that can be applied to various domains and tasks. Here are some potential use cases:

1. **Automated Research Assistant**: Create a pipeline that searches the web for a specific topic, summarizes the findings, and generates a comprehensive report with citations.

2. **Content Creation Workflow**: Design a pipeline that generates article ideas, writes draft content, creates accompanying images, and prepares social media posts for promotion.

3. **Sentiment Analysis and Brand Monitoring**: Set up a pipeline that scrapes social media and news sites for mentions of a brand, performs sentiment analysis, and generates daily reports with insights.

4. **Automated Customer Support**: Develop a pipeline that processes customer inquiries, searches a knowledge base for relevant information, and generates personalized responses.

5. **Data Analysis and Visualization**: Create a pipeline that collects data from various sources, performs statistical analysis, generates insights, and creates data visualizations.

6. **Language Translation Service**: Build a pipeline that takes text in one language, translates it to multiple target languages, and generates audio pronunciations for each translation.

7. **Automated Code Review**: Design a pipeline that analyzes code repositories, identifies potential issues or improvements, and generates detailed code review reports.

8. **Market Trend Analysis**: Set up a pipeline that monitors financial news, analyzes market data, and generates predictive reports on potential market trends.

9. **Educational Content Generation**: Create a pipeline that takes a subject area, generates educational content, quizzes, and supplementary materials for online courses.

10. **Personalized News Aggregator**: Develop a pipeline that collects news from various sources based on user preferences, summarizes articles, and generates a personalized daily news briefing.

These use cases demonstrate the flexibility and power of AIPE in automating complex workflows across different industries and applications. The modular nature of the pipeline allows for easy customization and expansion to suit specific needs.

## 🧩 Pipeline Steps

The AI Agent Pipeline Engine supports the following steps:

- `WebSearch`: Search the web using DuckDuckGo or Serper API
- `WebScrape`: Scrape content from specified URLs
- `RunInference`: Generate text using LLM models (Ollama, OpenAI)
- `TextToSpeech`: Convert text to speech
- `SpeechToText`: Transcribe speech to text
- `LoadImageModel`: Load a text-to-image model
- `GenerateImage`: Generate images from text prompts
- `CallWebhook`: Make external API calls

## 🛠️ Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/AIPE.git
   cd AIPE
   ```

2. Install the required dependencies:

   ```
   pip install -r app/requirements.txt
   ```

3. Set up environment variables for API keys (if needed):
   ```
   export OPENAI_API_KEY=your_openai_api_key
   export SERPER_API_KEY=your_serper_api_key
   ```

## 🚀 Usage

1. Create a YAML configuration file for your pipeline. Example:

   ```yaml
   pipeline:
    context:
        topic: "artificial intelligence with crypto"
        num_results: 5
        model_name: "llama2"

    steps:
        - name: "web_search"
        type: "WebSearch"
        query: "latest developments in {context.topic}"
        search_type: "serper"
        num_results: "{context.num_results}"

        - name: "summarize_findings"
        type: "RunInference"
        source: "ollama"
        model_name: "llama2"
        prompt: "Summarize the latest developments in {context.topic} based on the following information: {steps.web_search}"

        - name: "text_to_speech"
        type: "TextToSpeech"
        text: "{steps.summarize_findings}"
        result_path: "/app/output/summary_audio.wav"

        - name: "generate_image"
        type: "GenerateImage"
        prompt: "A futuristic representation of {context.topic}"
        result_path: "/app/output/ai_developments.png"
        image_params:
            num_inference_steps: 50
            guidance_scale: 7.5

        - name: "final_report"
        type: "RunInference"
        source: "ollama"
        model_name: "llama2"
        prompt: "Create a final report on {context.topic} incorporating the following elements:\n1. Web search results: {steps.web_search}\n2. Summarized findings: {steps.summarize_findings}\n4. Generated image description: A futuristic representation of {context.topic}\n5. Transcription accuracy check: Compare the original summary with the transcribed text and comment on any discrepancies."
        result_path: "/app/output/final_report.txt"
   ```

2. Run the pipeline:

   ```
   python app/main.py config/config.yaml
   ```

3. Check the output:
   The pipeline will save its outputs in the `output` folder. This includes generated files such as audio, images, and text reports as specified in your configuration.

## 🐳 Docker Support

The project includes Dockerfile for easy containerization. To build and run using Docker:

```
docker build -t AIPE .
docker run -it --gpus all --env-file .env AIPE
```

The pipeline will save its outputs in the `output` folder inside the docker container. This includes generated files such as audio, images, and text reports as specified in your configuration.

## 🌐 Spheron Deployment

To deploy your AIPE project on Spheron:

1. Update `spheron.yaml` in your project root with your image details.

2. Install `spheronctl` CLI:
   [Installation Guide](https://docs.spheron.network/user-guide/deploy-your-app#step-1-install-spheron-protocol-sphnctl-cli-linux-macos)

3. Create deployment:

   ```sh
   sphnctl deployment create spheron.yaml
   ```

4. Check deployment status (replace `<LID>` with your Deployment ID):

   ```sh
   sphnctl deployment get --lid <LID>
   ```

5. View deployment logs:

   ```sh
   sphnctl deployment logs --lid <LID>
   ```

6. Access deployment shell:
   ```sh
   sphnctl deployment shell aipet /bin/sh --lid <LID> --stdin --tty
   ```

For more details, see [Spheron Docs](https://docs.spheron.network/user-guide).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

If you have any questions, feel free to reach out to [Mitra](https://x.com/rekpero).
