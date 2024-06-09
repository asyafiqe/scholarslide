# ScholarSlide
## _Turn scientific paper into presentations, effortlessly._

Tired of spending hours summarizing research papers and creating slides? 
**ScholarSlide** uses AI to transform any scientific PDF into a clear and simple PowerPoint presentation. Simply upload your PDF, and ScholarSlide will automatically generate explanatory slides, ready for your next presentation.

## Features
- Summarize and make bullet points for each section of the paper (background, method, results, discussion, conclusion).
- Extract figures and tables and give explanation.
- Choose different model for text and vision.

- At current state, it is not presentation ready. It produce more slides than needed. You should hide some slide and choose which one is suitable. I think it is better to have more slide then remove it later than have to make new slide.

## Tech
- Backend: python
- Frontend: [streamsync](https://github.com/streamsync-cloud/streamsync)
- API: [OpenRouter](https://openrouter.ai/), [Adobe PDF Extract API](https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/)

## How to run
- Register to [OpenRouter](https://openrouter.ai/), click profile icon, Keys, Create Key, store it. May need to buy some credit.
- Get free [Adobe PDF Extract API](https://acrobatservices.adobe.com/dc-integration-creation-app-cdn/main.html?api=pdf-extract-api) key.
- Install [Docker](https://docs.docker.com/engine/install/)
- Run docker container
    ```sh
    docker run --env PDF_SERVICES_CLIENT_ID="your-api-key" --env PDF_SERVICES_CLIENT_SECRET="your-api-key" --env OPENROUTER_API_KEY="your-api-key" asyafiqe/scholarslide:latest
    ```
- Open localhost:5000 in your browser.


## License
Apache 2.0
