"""
Sample query pool covering all 20 ad categories.
Each query is a realistic user prompt that maps to at least one ad category.
"""

QUERIES = [
    # cloud
    "What's the easiest way to deploy a Python web app to the cloud?",
    "How do I host a Node.js backend without managing servers?",
    "I need auto-scaling for my app, what cloud platform should I use?",
    "My startup needs a simple cloud hosting solution with monitoring built in",
    "How do I deploy a Flask app with zero configuration?",

    # devops / CI-CD
    "How do I set up a CI/CD pipeline for my GitHub repo?",
    "What's the best way to automate deployments with GitLab?",
    "How do I roll back a bad deployment instantly?",
    "I want one-click deployments for my team, what tool should I use?",
    "How do I integrate automated testing into my deployment workflow?",

    # ml deployment
    "How do I serve a PyTorch model as a REST API?",
    "What's the fastest way to deploy a scikit-learn model to production?",
    "I trained a TensorFlow model, how do I host it with an endpoint?",
    "How do I build an ML inference API without managing infrastructure?",
    "My data science team needs to productionize models quickly",

    # ml pipeline / no-code
    "Is there a visual tool for building machine learning pipelines?",
    "How do I do automated hyperparameter tuning without writing code?",
    "I want to train a model with drag-and-drop feature engineering",
    "What no-code ML tools exist for non-engineers?",
    "How do I automate feature engineering for tabular data?",

    # vector database / embeddings
    "What is a vector database and when should I use one?",
    "How do I store and search embeddings for my LLM app?",
    "I need sub-10ms semantic search over millions of documents",
    "What's the best managed vector DB for a production LLM app?",
    "How do I do similarity search at scale for a recommendation system?",

    # auth / security
    "How do I add OAuth2 login to my app in a few minutes?",
    "What's the easiest way to implement MFA for my users?",
    "I need SAML and passwordless auth, is there a managed solution?",
    "How do I add social login (Google, GitHub) to my web app?",
    "My app needs SOC2 compliant authentication",

    # security scanning
    "How do I scan my codebase for security vulnerabilities automatically?",
    "Is there a tool that detects leaked secrets before I push to GitHub?",
    "How do I find SQL injection risks in my Python code?",
    "What tools exist for static security analysis in CI/CD?",
    "How do I prevent credentials from being committed to my repo?",

    # analytics / SQL
    "How do I connect my data warehouse to a dashboard tool?",
    "I want to run SQL queries on my data and build reports automatically",
    "What's a good analytics platform for my data team?",
    "How do I schedule SQL reports to send to my stakeholders?",
    "I need a self-serve analytics tool that connects to BigQuery and Snowflake",

    # streaming / real-time
    "How do I process millions of events per second in real time?",
    "What's a Kafka-compatible streaming platform I can use?",
    "I need real-time analytics on clickstream data",
    "How do I build a live dashboard on top of a Kafka stream?",
    "What's the best tool for real-time data pipelines?",

    # code review / devtools
    "Is there an AI tool that reviews my pull requests automatically?",
    "How do I catch bugs before code review using AI?",
    "What tools integrate with GitHub to give automated code feedback?",
    "I want AI-powered code review on every PR in my repo",
    "How do I set up automated code quality checks for my team?",

    # documentation
    "How do I auto-generate API docs from my codebase?",
    "Is there a tool that keeps OpenAPI docs in sync with my code?",
    "How do I generate GraphQL documentation automatically?",
    "I need API documentation that updates itself when I change the code",
    "What's the best way to document a gRPC API?",

    # logging / observability
    "How do I aggregate logs from distributed microservices?",
    "What's a good log search tool that handles terabytes of data?",
    "How do I trace requests across my distributed system?",
    "I need sub-second search over my application logs",
    "How do I set up alerting based on log patterns?",

    # LLM / prompt tooling
    "How do I version and A/B test my LLM prompts?",
    "Is there a tool to monitor OpenAI API costs per prompt?",
    "How do I debug why my GPT-4 responses are inconsistent?",
    "I want to track which prompt versions give the best results",
    "How do I compare prompt performance across different LLM providers?",

    # LLM app deployment
    "How do I deploy a LangChain app to production?",
    "What's the best hosting platform for a LlamaIndex RAG app?",
    "How do I scale an AI agent with auto-scaling?",
    "I need managed vector store infrastructure for my LLM pipeline",
    "How do I add agent tracing to my LangChain application?",

    # embeddings API
    "What's the cheapest embedding API for large-scale NLP?",
    "How do I get text embeddings for 50 languages at low cost?",
    "I need an OpenAI-compatible embeddings endpoint for my app",
    "What embedding model should I use for multilingual semantic search?",
    "How do I generate embeddings for a billion documents cheaply?",

    # education / ML learning
    "How do I learn machine learning from scratch with hands-on projects?",
    "What's a good course for deep learning with real datasets?",
    "I want to get a machine learning certificate to advance my career",
    "What's the best online resource for learning neural networks?",
    "How do I get started with deep learning as a software engineer?",

    # education / coding bootcamp
    "How do I become a full-stack developer in a few months?",
    "Is there a coding bootcamp that teaches Python, React, and cloud?",
    "I want to switch careers into software engineering, where do I start?",
    "What's a good program with a job placement guarantee for developers?",
    "How do I learn web development fast with hands-on projects?",

    # kubernetes / infra
    "How do I manage Kubernetes clusters without writing YAML?",
    "What's a visual tool for setting up Kubernetes auto-scaling?",
    "How do I optimize cloud costs for my Kubernetes workloads?",
    "I want to manage my K8s clusters visually without command-line tools",
    "How do I set up auto-scaling policies in Kubernetes easily?",

    # edge / CDN
    "How do I deploy my app globally with under 50ms latency?",
    "What's a CDN that includes WAF and DDoS protection?",
    "How do I serve my API from edge locations worldwide?",
    "I need global low-latency deployment for my mobile backend",
    "What's the best edge computing platform for a worldwide API?",
]
