"""
Creates a synthetic training dataset for fake job detection.
Run once: python -m data.create_dataset
"""
import os
import csv
import random

random.seed(42)

REAL_JOBS = [
    ("Software Engineer at Google. We are looking for an experienced software engineer to join our team. You will design and build the next generation of our systems. Requirements: 3+ years of experience in Python or Java, strong CS fundamentals, experience with distributed systems.", 0),
    ("Marketing Manager - ABC Corp. We are seeking a creative and data-driven marketing manager to lead our campaigns. Responsibilities include managing social media, planning campaigns, and analyzing performance data. Bachelor's degree required. Competitive salary and benefits.", 0),
    ("Data Scientist - FinTech Startup. Join our growing data science team. You will build predictive models for fraud detection and customer analytics. Skills required: Python, SQL, machine learning. Remote-friendly, health insurance provided.", 0),
    ("Registered Nurse - City Hospital. We are hiring registered nurses for ICU and general wards. Must have valid nursing license and 2+ years clinical experience. Full-time and part-time positions available. Excellent benefits package.", 0),
    ("Product Manager - SaaS Company. Looking for an experienced product manager to own our core product roadmap. You will work closely with engineering and design. 5+ years of PM experience required. Base salary + equity.", 0),
    ("Accountant - Mid-size firm. Seeking a qualified accountant to manage financial records, prepare tax returns, and conduct audits. CPA preferred. 3+ years experience required. Monday-Friday 9-5, office based.", 0),
    ("UI/UX Designer. We are looking for a talented designer with a strong portfolio. You will create wireframes, prototypes, and deliver pixel-perfect designs. Proficiency in Figma required. Salary: $85,000 - $105,000.", 0),
    ("DevOps Engineer - E-commerce Platform. Manage CI/CD pipelines, cloud infrastructure on AWS, and system reliability. Strong knowledge of Kubernetes and Terraform needed. Comprehensive benefits package including 401k.", 0),
    ("Sales Representative - B2B Software. Drive new business and manage accounts in the SMB segment. 2+ years sales experience preferred. Base + commission structure. Full training provided.", 0),
    ("Content Writer - Digital Agency. Create SEO-optimized blog posts and web content for clients in tech and finance. Must have portfolio. Freelance contract, flexible hours, competitive per-word rate.", 0),
    ("Operations Manager - Logistics Company. Oversee daily warehouse operations, manage staff, and optimize processes. 5+ years operations experience required. Salary commensurate with experience.", 0),
    ("Cybersecurity Analyst. Monitor security incidents, conduct vulnerability assessments, and respond to threats. CISSP or CEH certification preferred. Full-time, competitive compensation.", 0),
    ("Human Resources Specialist. Manage recruitment, onboarding, and employee relations. SHRM certification preferred. Strong communication skills required. Office-based role with hybrid option.", 0),
    ("Full Stack Developer - React/Node. Build and maintain web applications. Strong experience with React, Node.js, PostgreSQL. Agile environment, remote-friendly, annual salary review.", 0),
    ("Financial Analyst - Investment Bank. Analyze financial data, build models, and support deal teams. Excel and Bloomberg expertise required. MBA or CFA a plus.", 0),
    ("Customer Success Manager. Ensure client satisfaction and retention. Build long-term relationships with enterprise clients. 3+ years in SaaS customer success required.", 0),
    ("Electrical Engineer - Manufacturing. Design and maintain electrical systems and equipment. PE license preferred. On-site role with standard benefits and pension plan.", 0),
    ("School Teacher - Elementary. Teach grades 3-5. Valid teaching certificate required. Small class sizes, supportive administration. Summer off.", 0),
    ("Business Analyst - Consulting. Work with clients to document requirements and improve processes. Strong analytical and presentation skills. Travel up to 30%.", 0),
    ("Graphic Designer - Advertising. Create visual content for print and digital campaigns. Proficiency in Adobe Creative Suite. 3+ years agency experience preferred.", 0),
    ("Java Backend Developer. Design APIs and microservices for our banking platform. Spring Boot, Kafka, and AWS experience needed. Competitive salary, remote work allowed.", 0),
    ("Research Scientist - AI Lab. Publish novel research and build production ML systems. PhD in CS or related field preferred. Top-tier compensation package.", 0),
    ("Pharmacy Technician. Assist pharmacists in dispensing medications and serving customers. State certification required. Flexible shifts, competitive pay.", 0),
    ("Civil Engineer - Infrastructure. Oversee road and bridge projects from design to completion. PE license required, 5+ years experience. Strong project management skills.", 0),
    ("Social Media Manager. Plan, create, and analyze social media content across platforms. 2+ years social media experience. Hybrid work schedule.", 0),
]

FAKE_JOBS = [
    ("URGENT HIRING!!! Earn $5000 a week from home!!! No experience required!!! Send registration fee of $50 to get started. WhatsApp us now! Work from home, unlimited earnings guaranteed!!!", 1),
    ("Make money online from home. Copy paste job, earn $200 per day. No experience needed. Just pay a small registration fee of $99 and start earning immediately. Guaranteed income every week!", 1),
    ("Data entry work from home. Earn $300-$500 daily. No experience, no qualification needed. Just a smartphone. Send $30 as training fee via Western Union to receive your work kit.", 1),
    ("Part time work from home opportunity! Earn $150 per day just by filling simple online forms. Be your own boss! Payment via WhatsApp transfer. Urgent vacancies available. No interview required.", 1),
    ("WORK FROM HOME - Earn $1000 daily. Passive income opportunity. Free laptop and iPhone provided after registration. Pay $250 deposit refundable after 30 days. WhatsApp contact only.", 1),
    ("Online job vacancy! Earn $500/day working 2 hours from home. No experience needed, no age limit. Just send $75 registration fee via money order. Guaranteed payment every Monday.", 1),
    ("Home based job. Simple copy paste task, earn $400 per day. Investment required: $199 to activate your account. Be your own boss and earn unlimited. Start today!", 1),
    ("Urgently needed: 500 workers for online data entry. Earn $50 per hour from home. No experience required. Pay $49 joining fee to get access to work. WhatsApp interview only.", 1),
    ("Make $3000 weekly from home - part time from home job available. Send small fee of $80 for training materials. Earn money without any investment (after initial fee). Guaranteed income.", 1),
    ("Online typing job. Earn $200-$500 per day. No experience needed. Pay $59 upfront for starter kit via wire transfer. Work 1-2 hours daily. Unlimited earnings potential.", 1),
    ("Earn money from your phone! Just like and share posts on social media. Earn $100 per task. Registration fee of $30 required. Payment guaranteed daily. Work from home, no experience.", 1),
    ("GOVT APPROVED HOME JOB! Earn $2000 weekly. Fill forms from home. Investment required of $149 to start. No boss, no interview. WhatsApp for registration. Earn money now!", 1),
    ("Freelance data processing job. Earn $350 per day copying files. Training fee $99 to be sent via Western Union. No experience needed. Be your own boss. Urgent: 200 seats only!", 1),
    ("Work from home - earn money online. Simple tasks: watch videos, click ads. Earn $200/hour. Registration fee $25 required. Unlimited earnings, no experience. Guaranteed payment.", 1),
    ("Online assistant needed urgently. Work from home, earn $600/day. No experience, no interview. Pay $89 activation fee. Receive your first payment within 24 hours. WhatsApp contact.", 1),
    ("Passive income from home! Earn $5000/month doing nothing. Just refer friends and earn money. Registration fee $50. No experience. Be your own boss. Guaranteed returns!", 1),
    ("Data entry jobs available - no experience needed! Earn $1000 per week working from home. Just send $79 registration fee via wire transfer. Part time from home opportunity.", 1),
    ("Online work from home job! Earn $300 daily. Free training provided after $99 registration fee. No experience required. WhatsApp for immediate joining. Urgent vacancies!", 1),
    ("Home-based job. Copy paste simple text. Earn $250 per hour. No experience needed. Send $45 fee to unlock your account. Earn money guaranteed every day.", 1),
    ("LAST 5 SEATS! Earn $2500 weekly from home. No experience needed. Investment of $199 required upfront. Be your own boss. WhatsApp interview. Earn unlimited income!", 1),
    ("Online mystery shopper required urgently! Earn $500 per task. No experience needed. Background check fee of $60 required. Payment same day. WhatsApp for details.", 1),
    ("Social media evaluator - work from home! Earn $400 daily. Just check Facebook posts. Registration fee $35. No experience, no interview. Guaranteed weekly payment.", 1),
    ("Amazon home packer job! Pack orders from home. Earn $50/box. Send $99 kit fee. Work from home, no experience. Unlimited earnings. WhatsApp: immediate start!", 1),
    ("Earn money from home - survey jobs! Earn $200 per survey. Registration fee $29. No experience needed. Part time from home, flexible hours. Guaranteed payment!", 1),
    ("YouTube video rating job. Earn $100 per video watched. No experience required. Pay $55 activation fee. Work from home, earn unlimited. WhatsApp registration. Urgent!!", 1),
]

def augment(job, label, n=3):
    rows = [(job, label)]
    prefixes = [
        "HIRING NOW: ", "OPPORTUNITY: ", "JOB ALERT: ", "",
        "VACANCY: ", "NEW OPENING: "
    ]
    suffixes = [
        " Apply now!", " Limited seats.", " Contact us today.",
        " Don't miss this!", "", " Immediate joining."
    ]
    for _ in range(n):
        p = random.choice(prefixes)
        s = random.choice(suffixes)
        rows.append((p + job + s, label))
    return rows

rows = []
for job, label in REAL_JOBS + FAKE_JOBS:
    rows.extend(augment(job, label, n=6))

random.shuffle(rows)

out_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["description", "fraudulent"])
    writer.writerows(rows)

print(f"Created {len(rows)} samples -> {out_path}")
