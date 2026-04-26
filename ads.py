# Raw catalog data — no imports needed.
# pipeline.py reads this and constructs Ad objects.

AD_DATA = [
    # Cloud / DevOps
    dict(id="ad_01", name="CloudBase Pro",
         description="Deploy Python and Node apps to the cloud with one command. Auto-scaling, built-in monitoring, zero config.",
         category="cloud", target_cpa=45, daily_budget=400, base_ctr=0.045, base_cvr=0.12, avg_order_value=180),

    dict(id="ad_02", name="ServerlessNow",
         description="Run your backend without managing servers. Serverless functions with instant cold starts. Pay per use.",
         category="cloud", target_cpa=38, daily_budget=300, base_ctr=0.038, base_cvr=0.10, avg_order_value=150),

    dict(id="ad_03", name="GitOps Pipeline",
         description="Automate CI/CD workflows. One-click deployments, instant rollback, GitHub and GitLab integration.",
         category="devops", target_cpa=30, daily_budget=250, base_ctr=0.042, base_cvr=0.09, avg_order_value=120),

    # ML / AI
    dict(id="ad_04", name="ModelDeploy",
         description="Deploy machine learning models to production in minutes. REST API auto-generated. Supports PyTorch, TensorFlow, scikit-learn.",
         category="ml", target_cpa=62, daily_budget=600, base_ctr=0.052, base_cvr=0.14, avg_order_value=250),

    dict(id="ad_05", name="DataSpark ML",
         description="Visual ML pipeline builder. Drag-and-drop feature engineering, automated hyperparameter tuning, no-code model training.",
         category="ml", target_cpa=55, daily_budget=500, base_ctr=0.048, base_cvr=0.13, avg_order_value=220),

    dict(id="ad_06", name="VectorDB Cloud",
         description="Managed vector database for AI apps. Store and search billions of embeddings at sub-10ms latency. Built for LLMs.",
         category="ml", target_cpa=75, daily_budget=700, base_ctr=0.055, base_cvr=0.15, avg_order_value=300),

    # Security
    dict(id="ad_07", name="AuthShield",
         description="Add authentication to your app in 5 minutes. OAuth2, SAML, MFA, passwordless. SOC2 compliant.",
         category="security", target_cpa=40, daily_budget=350, base_ctr=0.040, base_cvr=0.11, avg_order_value=160),

    dict(id="ad_08", name="SecureScan",
         description="Automated security scanning for your codebase. Detect vulnerabilities and leaked secrets before they ship.",
         category="security", target_cpa=35, daily_budget=280, base_ctr=0.035, base_cvr=0.09, avg_order_value=140),

    # Data / Analytics
    dict(id="ad_09", name="QueryFlow",
         description="SQL analytics platform for data teams. Connect to any warehouse, build dashboards, schedule reports.",
         category="analytics", target_cpa=48, daily_budget=320, base_ctr=0.041, base_cvr=0.10, avg_order_value=190),

    dict(id="ad_10", name="StreamPulse",
         description="Real-time data streaming and analytics. Kafka-compatible, process millions of events per second.",
         category="analytics", target_cpa=52, daily_budget=450, base_ctr=0.047, base_cvr=0.12, avg_order_value=210),

    # Developer Tools
    dict(id="ad_11", name="CodeReview AI",
         description="AI-powered code review that catches bugs before your teammates do. GitHub, GitLab, Bitbucket integration.",
         category="devtools", target_cpa=25, daily_budget=220, base_ctr=0.043, base_cvr=0.11, avg_order_value=100),

    dict(id="ad_12", name="DocGen Pro",
         description="Auto-generate API documentation from your codebase. OpenAPI, GraphQL, gRPC. Always in sync with your code.",
         category="devtools", target_cpa=20, daily_budget=180, base_ctr=0.037, base_cvr=0.08, avg_order_value=80),

    dict(id="ad_13", name="LogSense",
         description="Intelligent log aggregation and alerting. Trace distributed requests, sub-1s search over terabytes of logs.",
         category="devtools", target_cpa=32, daily_budget=270, base_ctr=0.039, base_cvr=0.10, avg_order_value=130),

    # LLM / AI Apps
    dict(id="ad_14", name="PromptLayer",
         description="Track, version, and A/B test your LLM prompts. Monitor costs, debug responses. Works with OpenAI and Anthropic.",
         category="llm", target_cpa=70, daily_budget=550, base_ctr=0.058, base_cvr=0.16, avg_order_value=280),

    dict(id="ad_15", name="LangChain Cloud",
         description="Deploy LangChain and LlamaIndex apps to production. Managed vector stores, agent tracing, auto-scaling.",
         category="llm", target_cpa=80, daily_budget=650, base_ctr=0.060, base_cvr=0.17, avg_order_value=320),

    dict(id="ad_16", name="EmbedAPI",
         description="Fast, cheap text embeddings API. 1M tokens for $0.02. OpenAI-compatible format. 50+ languages.",
         category="llm", target_cpa=65, daily_budget=500, base_ctr=0.054, base_cvr=0.14, avg_order_value=260),

    # Education
    dict(id="ad_17", name="DeepLearn.io",
         description="Master machine learning and deep learning. 200+ hours of hands-on courses, real datasets, certificate included.",
         category="education", target_cpa=30, daily_budget=200, base_ctr=0.050, base_cvr=0.18, avg_order_value=120),

    dict(id="ad_18", name="CodeCamp Pro",
         description="Become a full-stack developer in 12 weeks. Python, React, databases, cloud. Job placement guarantee.",
         category="education", target_cpa=25, daily_budget=160, base_ctr=0.048, base_cvr=0.15, avg_order_value=100),

    # Infrastructure
    dict(id="ad_19", name="K8sEasy",
         description="Kubernetes made simple. Manage clusters visually, auto-scaling policies, cost optimization. No YAML required.",
         category="infra", target_cpa=50, daily_budget=380, base_ctr=0.044, base_cvr=0.11, avg_order_value=200),

    dict(id="ad_20", name="EdgeDeploy",
         description="Deploy to 200 edge locations worldwide. Sub-50ms global latency. DDoS protection and WAF included.",
         category="infra", target_cpa=42, daily_budget=340, base_ctr=0.040, base_cvr=0.10, avg_order_value=170),
]
