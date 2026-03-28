"""
Synthetic Resume Training Data Generator
Generates diverse resume -> summary pairs for fine-tuning language models with LoRA.
Covers multiple industries, experience levels, and formats.
Output: data/train.jsonl, data/val.jsonl
"""

import json
import os
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Raw resume templates
# ---------------------------------------------------------------------------

SAMPLES = [
    # --- Software Engineering ----------------------------------------------
    {
        "resume": """John Smith
Email: john.smith@email.com | Phone: 555-0101 | LinkedIn: linkedin.com/in/johnsmith
Location: San Francisco, CA

SUMMARY
Senior Software Engineer with 8 years of experience building scalable distributed systems.

EXPERIENCE
Senior Software Engineer - Google, Mountain View, CA (2019 – Present)
• Designed and implemented microservices architecture serving 50M+ daily users
• Led migration of monolithic application to Kubernetes, reducing deployment time by 70%
• Mentored team of 6 junior engineers; established code review standards
• Optimised database queries reducing p99 latency from 800ms to 120ms

Software Engineer - Airbnb, San Francisco, CA (2016 – 2019)
• Built real-time booking system processing 2M transactions/day
• Implemented A/B testing framework used by 12 product teams
• Contributed to open-source projects; 2,000+ GitHub stars

EDUCATION
B.S. Computer Science – Stanford University, 2016

SKILLS
Languages: Python, Go, Java, TypeScript
Frameworks: Kubernetes, gRPC, React, Django
Databases: PostgreSQL, Redis, Cassandra
Cloud: AWS, GCP""",
        "summary": "Senior Software Engineer with 8 years of experience at Google and Airbnb. Expert in distributed systems, microservices, and Kubernetes. Led 50M+ user platform migrations and mentored junior engineers. Proficient in Python, Go, Java, and TypeScript across AWS and GCP."
    },
    {
        "resume": """Priya Patel
priya.patel@dev.io | +1-415-555-0234 | GitHub: github.com/priyapatel

ABOUT
Full-stack developer specialising in React and Node.js. 4 years building SaaS products.

WORK HISTORY
Full Stack Developer - Stripe, Remote (2022 – Present)
• Built merchant dashboard used by 500,000+ businesses worldwide
• Reduced frontend bundle size by 45% through code splitting and lazy loading
• Implemented OAuth 2.0 integration with 20+ payment providers

Junior Developer - Shopify, Toronto (2020 – 2022)
• Developed Liquid theme components for 10,000+ merchant storefronts
• Fixed 200+ production bugs; improved test coverage from 60% to 90%

EDUCATION
Diploma in Software Development – BrainStation, 2020

TECH STACK
React, Next.js, Node.js, TypeScript, GraphQL, PostgreSQL, Docker""",
        "summary": "Full-stack developer with 4 years at Stripe and Shopify specialising in React and Node.js. Built merchant platforms serving 500K+ businesses and improved test coverage from 60% to 90%. Strong expertise in TypeScript, GraphQL, and Docker-based SaaS development."
    },
    {
        "resume": """Carlos Mendez
carlos@mendez.dev | San Jose, CA

EXPERIENCE
Staff Engineer - Meta (2021 – Present)
• Architect for WhatsApp Infrastructure team; systems handle 100B messages/day
• Designed sharding strategy reducing storage costs by $12M annually
• Patent holder: US Patent 11,234,567 - Distributed Message Routing System
• Presented at QCon 2023 on scalable messaging architectures

Senior Engineer - Twitter (2017 – 2021)
• Core contributor to Finagle RPC framework (14K GitHub stars)
• Built timeline ranking service serving 200M users

Software Engineer - LinkedIn (2014 – 2017)
• Developed LinkedIn Feed relevance algorithms

EDUCATION
M.S. Computer Science – UC Berkeley, 2014
B.S. Computer Engineering – UCLA, 2012

SKILLS
Scala, Java, C++, Rust | Distributed Systems, Consensus Protocols, Systems Design""",
        "summary": "Staff Engineer at Meta with 10+ years across Meta, Twitter, and LinkedIn. Architect of WhatsApp infrastructure processing 100B messages/day; drove $12M annual storage savings. Patent holder in distributed systems. M.S. Computer Science, UC Berkeley. Expert in Scala, Java, Rust, and large-scale distributed systems."
    },

    # --- Data Science / ML -------------------------------------------------
    {
        "resume": """Dr. Aisha Johnson
aisha.johnson@ml.com | New York, NY | Scholar: scholar.google.com/aishajohnson

EXPERIENCE
Principal Data Scientist - JPMorgan Chase (2020 – Present)
• Built fraud detection model saving $340M annually; 99.2% precision
• Led team of 8 data scientists; managed $4M ML infrastructure budget
• Deployed real-time ML pipeline processing 5M transactions/hour on AWS SageMaker

Senior Data Scientist - Amazon (2017 – 2020)
• Developed product recommendation engine boosting revenue by 18%
• Designed demand forecasting model reducing inventory waste by $200M

EDUCATION
Ph.D. Statistics – Columbia University, 2017
B.S. Mathematics – MIT, 2013

PUBLICATIONS
• Johnson et al., "Robust Fraud Detection in High-Frequency Transactions", NeurIPS 2022
• Johnson et al., "Temporal Graph Networks for E-Commerce", KDD 2019

SKILLS
Python, R, Spark, TensorFlow, PyTorch, SQL, AWS SageMaker, Databricks""",
        "summary": "Principal Data Scientist at JPMorgan Chase with Ph.D. in Statistics from Columbia. Built fraud detection model saving $340M/year at 99.2% precision. Led 8-person ML team and managed $4M budget. NeurIPS and KDD publications. Expert in Python, PyTorch, Spark, and AWS SageMaker."
    },
    {
        "resume": """Liang Wei
liang.wei@datascience.com | Seattle, WA

PROFESSIONAL EXPERIENCE
Machine Learning Engineer - Microsoft (2021 – Present)
• Fine-tuned large language models for Copilot features used by 1M+ enterprise users
• Reduced model inference latency by 60% via quantisation and ONNX optimisation
• Collaborated with Azure OpenAI team on RAG pipeline architecture

Data Scientist - Expedia (2018 – 2021)
• Built hotel price prediction model with 92% accuracy
• Developed NLP pipeline for review sentiment analysis (10M reviews/month)

EDUCATION
M.S. Machine Learning – Carnegie Mellon University, 2018

SKILLS
PyTorch, HuggingFace Transformers, ONNX, Azure ML, Python, Spark, SQL
LLM fine-tuning, RAG, RLHF, Vector Search""",
        "summary": "Machine Learning Engineer at Microsoft with M.S. from Carnegie Mellon. Fine-tuned LLMs for Copilot serving 1M+ enterprise users; cut inference latency 60% via quantisation. Expert in PyTorch, HuggingFace Transformers, RAG, and Azure ML. Prior data science experience at Expedia."
    },

    # --- Cybersecurity -----------------------------------------------------
    {
        "resume": """Marcus Thompson
marcus.thompson@security.com | Washington, DC
Certifications: CISSP, CEH, OSCP, AWS Security Specialty

EXPERIENCE
Principal Security Engineer - CrowdStrike (2020 – Present)
• Architected zero-trust network for 5,000-employee enterprise
• Led red team exercises identifying 47 critical vulnerabilities
• Built threat intelligence platform ingesting 10M IOCs/day
• Authored internal playbooks for SOC incident response

Security Engineer - Palo Alto Networks (2017 – 2020)
• Developed detection rules for WildFire sandbox (blocked 5M malware samples)
• Reverse-engineered 200+ malware samples; published 12 threat research blogs

EDUCATION
B.S. Information Security – George Mason University, 2017

SKILLS
Penetration Testing, Threat Hunting, Malware Analysis, IDS/IPS, SIEM
Python, PowerShell, Go | AWS, Azure Security Center""",
        "summary": "Principal Security Engineer at CrowdStrike with CISSP, CEH, and OSCP certifications. Architected zero-trust networks, led red team exercises uncovering 47 critical vulnerabilities, and built threat intel platforms at enterprise scale. Published malware researcher with deep expertise in penetration testing, Python, and cloud security."
    },
    {
        "resume": """Sarah Chen
sarah.chen@infosec.io | Austin, TX
Certs: CISM, CompTIA Security+, Azure Security Engineer Associate

EXPERIENCE
Information Security Manager - Dell Technologies (2019 – Present)
• Manage 12-person SOC team monitoring 200,000 endpoints
• Reduced mean time to detect (MTTD) from 72 hours to 4 hours
• Implemented ISO 27001 programme; achieved certification in 8 months
• $8M annual security budget ownership

Security Analyst III - Deloitte (2015 – 2019)
• Conducted 50+ security assessments for Fortune 500 clients
• Developed NIST CSF compliance roadmaps for healthcare clients

EDUCATION
M.S. Cybersecurity – University of Texas, 2015

SKILLS
SIEM (Splunk, QRadar), EDR, Vulnerability Management, GRC, ISO 27001, NIST""",
        "summary": "Information Security Manager at Dell Technologies with M.S. in Cybersecurity and CISM certification. Leads 12-person SOC monitoring 200K endpoints; reduced MTTD from 72 hours to 4 hours. Delivered ISO 27001 certification and manages $8M security budget. Extensive GRC and NIST compliance background from Deloitte."
    },

    # --- Product Management ------------------------------------------------
    {
        "resume": """Rachel Kim
rachel.kim@pm.com | New York, NY

EXPERIENCE
Senior Product Manager - Uber (2021 – Present)
• Owned Driver Earnings product; launched features increasing driver income by 22%
• Defined roadmap for Uber Pro loyalty programme (4M active members)
• Partnered with engineering, design, and data science teams of 40+

Product Manager - Lyft (2018 – 2021)
• Launched Express Drive rental programme in 15 cities
• A/B tested 60+ pricing experiments; improved conversion rate by 34%

Associate PM - Google (2016 – 2018)
• Contributed to Google Maps transit features used by 1B+ users

EDUCATION
MBA – Wharton School, University of Pennsylvania, 2016
B.A. Economics – Duke University, 2014

SKILLS
Product Strategy, Roadmapping, SQL, Tableau, Mixpanel, JIRA, Agile/Scrum""",
        "summary": "Senior Product Manager at Uber with MBA from Wharton. Owns Driver Earnings product driving 22% income growth and Uber Pro loyalty programme with 4M members. Track record of data-driven product development across Uber, Lyft, and Google Maps. Skilled in SQL, Tableau, and Agile execution."
    },
    {
        "resume": """David Okonkwo
d.okonkwo@product.com | Chicago, IL

SUMMARY
Experienced Product leader in B2B SaaS with focus on enterprise workflow automation.

EXPERIENCE
Director of Product - Salesforce (2022 – Present)
• Lead product vision for Salesforce Flow (3M automation users)
• Grew ARR from $80M to $210M over 2 years through new enterprise features
• Built and scaled product team from 4 to 14 PMs

Senior PM - ServiceNow (2019 – 2022)
• Launched IT Asset Management module adopted by 800 enterprise clients
• Reduced customer onboarding time from 6 weeks to 2 weeks

EDUCATION
B.S. Industrial Engineering – Northwestern University, 2013

SKILLS
Enterprise SaaS, Go-to-Market Strategy, OKRs, Customer Discovery, SQL, Looker""",
        "summary": "Director of Product at Salesforce leading Salesforce Flow with 3M users; grew product ARR from $80M to $210M in 2 years. Scaled PM team from 4 to 14. Prior enterprise product leadership at ServiceNow. Northwestern Industrial Engineering graduate with deep B2B SaaS and go-to-market expertise."
    },

    # --- Finance / Accounting ----------------------------------------------
    {
        "resume": """Emily Rodriguez
emily.rodriguez@finance.com | New York, NY
CFA Charterholder | CPA

EXPERIENCE
Vice President, Investment Banking – Goldman Sachs (2018 – Present)
• Executed 25+ M&A transactions totalling $14B in deal value
• Led IPO of TechCorp ($2.1B, NASDAQ 2022) and SpaceCo ($800M, NYSE 2023)
• Managed relationships with 40+ C-suite clients

Associate – Morgan Stanley (2015 – 2018)
• Built LBO, DCF, and comparable company models for TMT sector
• Supported $3.5B leveraged buyout of SoftwareCo

EDUCATION
MBA (Finance) – Columbia Business School, 2015
B.S. Finance – NYU Stern, 2013

SKILLS
M&A, Equity Capital Markets, Financial Modelling, Bloomberg, FactSet, Excel, VBA""",
        "summary": "Vice President in Investment Banking at Goldman Sachs with CFA and CPA credentials. Closed 25+ M&A transactions worth $14B and led two public offerings raising $2.9B. MBA from Columbia Business School. Expert in LBO/DCF modelling, equity capital markets, and C-suite client management."
    },

    # --- Healthcare / Medicine ---------------------------------------------
    {
        "resume": """Dr. James Osei
james.osei@hospital.org | Boston, MA
Licenses: MD, Board Certified Internal Medicine, ABIM

EXPERIENCE
Attending Physician, Internal Medicine – Massachusetts General Hospital (2018 – Present)
• Manage caseload of 20+ inpatients daily; 98% patient satisfaction score
• Clinical lead for hospital's diabetes management programme (1,200 enrolled patients)
• Supervise 8 residents and 4 medical students per rotation

Resident, Internal Medicine – Johns Hopkins Hospital (2015 – 2018)
• Chief Resident, PGY-3; awarded Outstanding Resident Award 2018

EDUCATION
M.D. – Harvard Medical School, 2015
B.S. Biology – Yale University, 2011

RESEARCH
• 14 peer-reviewed publications; 450+ citations
• NIH R01 grant recipient ($1.2M, 2021-2025)""",
        "summary": "Board-certified Internist at Massachusetts General Hospital with M.D. from Harvard Medical School. Leads diabetes management programme for 1,200 patients and supervises residents daily. Chief Resident alumnus at Johns Hopkins. Active NIH-funded researcher with 14 publications and $1.2M R01 grant."
    },

    # --- Marketing ---------------------------------------------------------
    {
        "resume": """Natalie Brooks
natalie.brooks@marketing.com | Los Angeles, CA

EXPERIENCE
VP of Marketing – Spotify (2021 – Present)
• Led global campaign for Spotify Wrapped reaching 400M users in 184 countries
• Grew podcast advertising revenue from $200M to $550M in 2 years
• Manage $120M annual marketing budget and team of 35

Marketing Director – Netflix (2017 – 2021)
• Launched marketing for 50+ original series including top-10 global hits
• Developed influencer strategy generating 2B social media impressions

EDUCATION
M.S. Marketing – Northwestern (Kellogg), 2017
B.A. Communications – USC, 2015

SKILLS
Brand Strategy, Performance Marketing, SEO/SEM, Google Analytics, Tableau, Salesforce""",
        "summary": "VP of Marketing at Spotify managing $120M budget and 35-person team. Drove Spotify Wrapped to 400M users globally and grew podcast ad revenue from $200M to $550M. M.S. Marketing from Kellogg. Expert in brand strategy, performance marketing, and data-driven campaign management from Spotify and Netflix."
    },

    # --- Entry Level -------------------------------------------------------
    {
        "resume": """Alex Turner
alex.turner@student.edu | Boston, MA | github.com/alexturner

EDUCATION
B.S. Computer Science – Boston University (Expected May 2024)
GPA: 3.8/4.0 | Dean's List

PROJECTS
Personal Finance App (React Native, Firebase)
• Built cross-platform app with 500+ downloads on App Store
• Implemented ML-based spending categorisation with 89% accuracy

Sentiment Analysis Tool (Python, BERT)
• Trained BERT model on 50K tweets; achieved 91% F1 score
• Published on GitHub with 120+ stars

INTERNSHIPS
Software Engineering Intern – HubSpot, Boston (Summer 2023)
• Developed REST API endpoints serving 10K daily requests
• Resolved 30 bugs reducing support tickets by 15%

SKILLS
Python, Java, React, React Native, SQL, Git, AWS (S3, Lambda)""",
        "summary": "Final-year Computer Science student at Boston University (GPA 3.8) with software engineering internship at HubSpot. Built App Store-published React Native app with ML spending categorisation. Proficient in Python, Java, React, and AWS. Strong ML fundamentals demonstrated through BERT sentiment project with 91% F1 score."
    },
    {
        "resume": """Sofia Martinez
sofia.m@email.com | Miami, FL

EDUCATION
B.B.A. Marketing – University of Miami (2023)
GPA: 3.6 | Marketing Club President

INTERNSHIPS
Digital Marketing Intern – Chewy, Dania Beach (Summer 2022)
• Managed Google Ads campaigns with $50K monthly budget; improved ROAS by 28%
• Created 15 email campaigns achieving 35% average open rate

Content Marketing Intern – HubSpot Blog, Remote (Spring 2022)
• Wrote 20+ SEO-optimised blog posts driving 80K organic visits
• Assisted with social media content calendar for 500K followers

SKILLS
Google Ads, Facebook Ads, HubSpot, Mailchimp, Canva, Google Analytics, Excel
Bilingual: English and Spanish""",
        "summary": "Recent marketing graduate from University of Miami with hands-on paid and content marketing internship experience. Managed $50K/month Google Ads budget at Chewy with 28% ROAS improvement. Generated 80K organic visits through SEO content at HubSpot. Proficient in Google Ads, HubSpot, and analytics tools. Bilingual in English and Spanish."
    },

    # --- Operations / Supply Chain -----------------------------------------
    {
        "resume": """Michael Chen
m.chen@ops.com | Detroit, MI
Certifications: PMP, Six Sigma Black Belt, APICS CSCP

EXPERIENCE
Director of Supply Chain – Ford Motor Company (2019 – Present)
• Manage $2.4B annual procurement budget across 350 global suppliers
• Implemented supplier risk programme reducing disruptions by 40%
• Led cross-functional team of 80 during COVID-19 semiconductor shortage response
• Saved $180M through strategic sourcing and contract renegotiation

Supply Chain Manager – Toyota (2014 – 2019)
• Implemented lean manufacturing principles; reduced waste by 32%
• Managed just-in-time inventory for 12 production lines

EDUCATION
M.S. Supply Chain Management – Michigan State, 2014
B.S. Industrial Engineering – Purdue, 2012

SKILLS
SAP, Oracle SCM, Tableau, Supplier Negotiation, Lean/Six Sigma, S&OP""",
        "summary": "Director of Supply Chain at Ford Motor Company managing $2.4B procurement budget across 350 global suppliers. PMP and Six Sigma Black Belt certified. Saved $180M through strategic sourcing and led COVID-19 semiconductor shortage response for 80-person cross-functional team. M.S. Supply Chain from Michigan State; expert in SAP, Lean, and S&OP."
    },

    # --- Human Resources ---------------------------------------------------
    {
        "resume": """Jennifer Walsh
j.walsh@hr.com | Chicago, IL
Certifications: SHRM-SCP, PHR

EXPERIENCE
Chief People Officer – Groupon (2020 – Present)
• Lead HR function for 3,000-employee global organisation across 15 countries
• Reduced voluntary attrition from 28% to 14% through culture and compensation redesign
• Built DEI programme increasing underrepresented group representation by 18%
• Drove HR digital transformation; implemented Workday HCM for 3,000 employees

VP, Human Resources – United Airlines (2016 – 2020)
• Negotiated 3 union contracts covering 20,000 employees
• Launched manager effectiveness programme improving engagement scores by 22 pts

EDUCATION
M.S. HR Management – Cornell ILR School, 2010

SKILLS
Talent Acquisition, Compensation & Benefits, HRIS (Workday, SAP SuccessFactors), ER, DEI""",
        "summary": "Chief People Officer at Groupon leading HR for 3,000 employees across 15 countries. SHRM-SCP certified with M.S. from Cornell ILR. Reduced attrition from 28% to 14%, built measurable DEI programme, and implemented Workday HCM. Negotiated union contracts covering 20,000 employees at United Airlines. Expert in talent strategy, compensation, and HR digital transformation."
    }
]

# ---------------------------------------------------------------------------
# Prompt template (matches fine-tuning format used in train.py)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "### Task: Summarise the following resume into 2-3 concise sentences covering "
    "the candidate's current role, years of experience, key achievements, and core skills.\n\n"
    "### Resume:\n{resume}\n\n"
    "### Summary:\n"
)


def build_record(sample: dict) -> dict:
    """Build a single training record with prompt + completion."""
    return {
        "prompt":     PROMPT_TEMPLATE.format(resume=sample["resume"].strip()),
        "completion": sample["summary"].strip(),
        "text":       PROMPT_TEMPLATE.format(resume=sample["resume"].strip()) + sample["summary"].strip(),
    }


def generate(output_dir: str = "data", val_split: float = 0.15):
    os.makedirs(output_dir, exist_ok=True)

    records = [build_record(s) for s in SAMPLES]
    random.shuffle(records)

    n_val   = max(1, int(len(records) * val_split))
    n_train = len(records) - n_val

    train_records = records[:n_train]
    val_records   = records[n_train:]

    def _write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f" Written {len(data):>3} records -> {path}")

    _write(os.path.join(output_dir, "train.jsonl"), train_records)
    _write(os.path.join(output_dir, "val.jsonl"),   val_records)

    print(f"\nDataset summary: {n_train} train / {n_val} val (total {len(records)})")
    return train_records, val_records


if __name__ == "__main__":
    print("Generating synthetic resume training data ...")
    generate()