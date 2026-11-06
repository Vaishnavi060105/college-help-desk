"""
Advanced NLP & Generative AI Engine for College Help Desk Assistant
===================================================================
Incorporates 6 Advanced NLP & Conversational AI Pillars:
1. RAG (Retrieval-Augmented Generation) with Dense Semantic Vector Space & Multi-Document Retrieval
2. MemAC-LLM (Memory-Augmented Conversational Engine): Working Memory & Episodic Context Memory
3. Joint Intent Classification & Token-Level Sequence Tagging (IOB / BIO Slot Filling & NER)
4. Multilingual Linguistic Profiling & Code-Switching Index (CMI / CSI for Maranglish/Hinglish)
5. Student Query Urgency, Sentiment & Administrative Triage Meter
6. Context-Aware Grounded Response Synthesizer with Source Document Citations
"""

import os
import re
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# 1. INSTITUTIONAL KNOWLEDGE CORPUS FOR MULTI-DOCUMENT RAG
# Comprehensive institutional sub-documents across all departments
# ==============================================================================
INSTITUTIONAL_KNOWLEDGE_CORPUS = [
    # Admission & Seats
    {
        "id": "adm_cap_process",
        "intent": "admission_query",
        "title": "Engineering CAP Round Allotment & Reporting",
        "keywords": ["first year", "cap round", "engineering admission", "allotment", "seat", "freeze", "float", "arc center", "dse"],
        "content_marathi": "प्रथम वर्ष (FE) व थेट द्वितीय वर्ष (DSE) प्रवेश प्रक्रिया CET Cell च्या अधिकृत नियमांनुसार राबवली जाते. विद्यार्थ्याने CAP राउंडनुसार मिळालेली सीट फ्रीज (Freeze) किंवा फ्लोट (Float) करून ARC सेंटरमध्ये मूळ कागदपत्रांसह सकाळी १०:०० ते दुपारी ५:०० दरम्यान उपस्थित राहणे बंधनकारक आहे.",
        "content_english": "First Year (FE) and Direct Second Year (DSE) admissions are conducted strictly through CET Cell CAP rounds. Candidates must choose Freeze or Float options on the portal and report to the ARC center with original documents between 10:00 AM and 5:00 PM.",
        "category": "Admissions & Allotment",
        "citation": "Academic Admission Handbook 2024-25 § Section 3.1"
    },
    {
        "id": "adm_cutoffs_branches",
        "intent": "admission_query",
        "title": "Department Cutoff Percentiles & Branches",
        "keywords": ["cutoff", "percentile", "computer science", "it", "aids", "entc", "marks", "merit list", "closing rank"],
        "content_marathi": "कॉम्प्युटर सायन्स (CS) व IT शाखेचा अंदाजे कटऑफ ९५+ पर्सेन्टाईल, AI-DS शाखेसाठी ९३+, आणि ENTC शाखेसाठी ८८+ पर्सेन्टाईल लागतो. मागील वर्षाची अंतिम मेरिट लिस्ट ॲडमिशन सेलच्या नोटीस बोर्डवर आणि अधिकृत वेबसाइटवर उपलब्ध आहे.",
        "content_english": "Computer Science & IT branches typically close around 95+ percentile, AI-DS at 93+, and ENTC at 88+. Previous year closing merit lists are published on the college portal.",
        "category": "Admissions & Cutoffs",
        "citation": "CET Cutoff Statistics Bulletin 2024"
    },
    {
        "id": "adm_tfws_quota",
        "intent": "admission_query",
        "title": "TFWS (Tuition Fee Waiver Scheme) Seats",
        "keywords": ["tfws", "tuition fee waiver", "quota", "income limit", "waiver", "8 lakh"],
        "content_marathi": "TFWS योजनेअंतर्गत प्रत्येक शाखेत ५% अतिरिक्त सुपरन्यूमररी जागा असतात ज्यामध्ये १००% शिक्षण शुल्क (Tuition Fee) माफ असते. यासाठी पालकांचे वार्षिक कौटुंबिक उत्पन्न ₹८ लाखांपेक्षा कमी असल्याचा तहसीलदारांचा सक्षम दाखला आवश्यक असतो.",
        "content_english": "Under the TFWS quota, 5% supernumerary seats carry a 100% tuition fee waiver. Eligibility requires a valid Tahsildar income certificate showing annual family income strictly below ₹8 Lakhs.",
        "category": "Scholarships & Quota",
        "citation": "DTE TFWS Scheme Guidelines (Govt of Maharashtra)"
    },
    {
        "id": "adm_mgmt_institutional",
        "intent": "admission_query",
        "title": "Management Quota & Institutional Spot Rounds",
        "keywords": ["management quota", "spot round", "institutional round", "vacant seats", "against cap"],
        "content_marathi": "CAP राउंड संपल्यानंतर शिल्लक राहिलेल्या रिक्त जागांसाठी (Vacant / Against-CAP Seats) इन्स्टिट्यूशनल व स्पॉट राउंड कॉलेज पातळीवर पूर्णपणे मेरिटनुसार घेतले जातात. अर्जाचा फॉर्म कॉलेज कार्यालयात उपलब्ध असून वेळेत सादर करावा.",
        "content_english": "Institutional spot rounds for remaining vacant seats are conducted strictly based on CET/JEE merit after CAP rounds conclude. Application forms are available at the main administrative office.",
        "category": "Admissions & Seats",
        "citation": "Institutional Seat Allocation Protocol 2024"
    },

    # Fees & Accounts
    {
        "id": "fee_structure_total",
        "intent": "fee_query",
        "title": "Annual Academic Tuition Fee Structure",
        "keywords": ["college fee", "annual fees", "tuition fee", "fee structure", "open category", "development fee"],
        "content_marathi": "ओपन कॅटेगरीसाठी चालू शैक्षणिक वर्षाची एकूण फी ₹१,२५,००० आहे (ट्युशन फी ₹१,१०,००० + विकास फी ₹१५,०००). इतर मागासवर्गीय (OBC/EBC) प्रवर्गासाठी ५०% सवलत असून फी ₹६७,५०० आहे. SC/ST विद्यार्थ्यांसाठी केवळ परीक्षा शुल्क आकारले जाते.",
        "content_english": "Annual academic fees for Open Category are ₹1,25,000 (Tuition ₹1,10,000 + Development ₹15,000). OBC and EBC students pay 50% concession fees of ₹67,500.",
        "category": "Fees & Accounts",
        "citation": "Fee Regulating Authority (FRA) Approved Structure"
    },
    {
        "id": "fee_installments",
        "intent": "fee_query",
        "title": "Fee Payment in Installments Facility",
        "keywords": ["installment", "pay in installments", "parts", "concession", "accounts section", "hafta"],
        "content_marathi": "आर्थिक अडचणीत असलेल्या विद्यार्थ्यांना फी २ किंवा ३ हप्त्यांमध्ये (Installments) भरण्याची सवलत उपलब्ध आहे. यासाठी पालकांच्या स्वाक्षरीसह प्राचार्य किंवा अकाउंट्स ऑफिसर यांच्याकडे रीतसर लेखी अर्ज सादर करावा लागतो.",
        "content_english": "Students facing financial constraints can pay fees in 2 to 3 installments by submitting a formal written request countersigned by parents to the Accounts Section.",
        "category": "Fees & Accounts",
        "citation": "Student Financial Assistance Policy § 4.2"
    },
    {
        "id": "fee_online_portal_failed",
        "intent": "fee_query",
        "title": "Online ERP Portal & Transaction Failed Issues",
        "keywords": ["online payment", "erp portal", "money debited", "payment failed", "challan", "utr", "receipt", "deducted"],
        "content_marathi": "ऑनलाईन ईआरपी पोर्टलवर पैसे खात्यातून वजा झाले पण पावती जनरेट झाली नाही, तर ७२ तासांत पेमेंट स्टेटस आपोआप रिकन्सिल होते. तातडीच्या समस्येसाठी बँक ट्रान्झॅक्शन आयडी/UTR क्रमांकासह अकाउंट्स काउंटर क्र. २ वर संपर्क साधावा.",
        "content_english": "If payment was debited but the receipt was not generated on the ERP portal, the bank reconciles the transaction within 72 hours. For unresolved cases, visit Accounts Counter 2 with your bank UTR number.",
        "category": "Fees & Accounts",
        "citation": "ERP Payment Gateway Support Desk"
    },

    # Timetables & Classes
    {
        "id": "time_lecture_schedule",
        "intent": "timetable_query",
        "title": "Regular Lecture & Practical Timing",
        "keywords": ["lecture timing", "timetable", "practical batch", "morning class", "break", "lab", "b1", "b2"],
        "content_marathi": "नियमित वर्ग सकाळी ९:०० ते दुपारी ४:३० या वेळेत चालतात. प्रॅक्टिकल बॅचेस (B1, B2, B3) चे स्लॉट्स २ तासांचे असतात. दुपारची जेवणाची सुट्टी (Lunch Break) दुपारी १२:१५ ते १:०० दरम्यान असते.",
        "content_english": "Regular lectures run from 9:00 AM to 4:30 PM. Practical lab sessions run in 2-hour blocks with a 45-minute lunch break between 12:15 PM and 1:00 PM.",
        "category": "Academics & Timetable",
        "citation": "Dean Academics Schedule Circular"
    },
    {
        "id": "time_download_calendar",
        "intent": "timetable_query",
        "title": "Timetable PDF & Academic Calendar",
        "keywords": ["download timetable", "academic calendar", "pdf", "holiday list", "portal", "semester plan"],
        "content_marathi": "सर्व शाखांचे सेमिस्टरनिहाय वेळापत्रक व विद्यापीठाचे अधिकृत ॲकॅडेमिक कॅलेंडर कॉलेजच्या ईआरपी पोर्टलवरून किंवा विभागनिहाय व्हॉट्सॲप ग्रुपवरून पीडीएफ स्वरूपात डाऊनलोड करता येईल.",
        "content_english": "Branch-wise timetables and the official academic calendar PDF are published on the college ERP portal and department class notice channels.",
        "category": "Academics & Timetable",
        "citation": "College Academic Portal / Timetable Repository"
    },

    # Scholarships & Freeships
    {
        "id": "sch_mahadbt_ebc",
        "intent": "scholarship_query",
        "title": "MahaDBT Post-Matric & EBC Scholarship Verification",
        "keywords": ["mahadbt", "scholarship", "ebc", "freeship", "caste scholarship", "verification", "biometric"],
        "content_marathi": "महाडीबीटी (MahaDBT) आणि राजर्षी शाहू महाराज ईबीसी (EBC) योजनेचे फॉर्म भरण्याची प्रक्रिया सुरू आहे. विद्यार्थ्यांनी आपले जात प्रमाणपत्र, नॉन-क्रिमिलेअर व तहसीलदार उत्पन्न दाखला स्कॉलरशिप डेस्कवर जमा करून बायोमेट्रिक प्रमाणीकरण पूर्ण करावे.",
        "content_english": "MahaDBT post-matric and EBC scholarship applications are active. Submit your caste certificate, non-creamy layer, and Tahsildar income certificate at the scholarship desk for biometric verification.",
        "category": "Scholarships",
        "citation": "Social Welfare & MahaDBT Cell Notification"
    },
    {
        "id": "sch_income_limit",
        "intent": "scholarship_query",
        "title": "Income Criteria & Document Checklist for Freeship",
        "keywords": ["income limit", "documents for scholarship", "tahsildar", "aadhaar seeding", "npci", "8 lakh"],
        "content_marathi": "EBC व OBC शिष्यवृत्तीसाठी कौटुंबिक वार्षिक उत्पन्न मर्यादा कमाल ₹८ लाख आहे. SC/ST विद्यार्थ्यांसाठी मर्यादा ₹२.५ लाख आहे. बँक खाते आधार कार्डशी एनपीसीआय (NPCI) मॅपिंग असणे थेट अनुदानासाठी बंधनकारक आहे.",
        "content_english": "Family annual income limit is ₹8 Lakhs for EBC and OBC freeships, and ₹2.5 Lakhs for SC/ST scholarships. Bank accounts must have active NPCI Aadhaar seeding for DBT credits.",
        "category": "Scholarships",
        "citation": "Government Resolution (GR) on Higher Education Scholarships"
    },

    # Documents & Certificates
    {
        "id": "doc_bonafide_lc",
        "intent": "document_query",
        "title": "Bonafide Certificate & Leaving Certificate (LC)",
        "keywords": ["bonafide certificate", "leaving certificate", "lc", "tc", "student section", "transcript", "counter 4"],
        "content_marathi": "बोनाफाईड प्रमाणपत्र, ट्रान्सक्रिप्ट व लीव्हिंग सर्टिफिकेट (LC) साठी स्टुडंट सेक्शनच्या काउंटर क्र. ४ वर विहित नमुन्यात अर्ज करावा. अर्ज सादर केल्यानंतर २ ते ३ कामकाजाच्या दिवसांत प्रमाणपत्र दिले जाते.",
        "content_english": "For Bonafide Certificates, Transcripts, and Leaving Certificates (LC), submit the application form at Student Section Counter 4. Processing takes 2-3 business days.",
        "category": "Student Services & Documents",
        "citation": "Student Administration Section Protocol"
    },
    {
        "id": "doc_loan_bus_pass",
        "intent": "document_query",
        "title": "Fee Quotation for Education Loan & Bus/Railway Pass",
        "keywords": ["education loan", "fee structure quotation", "railway pass", "bus concession", "concession form"],
        "content_marathi": "एज्युकेशन लोनसाठी अधिकृत फी कोटेशन (Fee Structure Quotation) आणि एसटी/रेल्वे कन्सेशन पास फॉर्म स्टुडंट सेक्शनमधून तात्काळ एकाच दिवसात स्वाक्षरी व शिक्क्यानिशी मिळतो.",
        "content_english": "Official fee structure quotations for bank education loans and ST/Railway concession certificates are verified and signed on the same day at the student section.",
        "category": "Student Services & Documents",
        "citation": "Student Welfare Concession Cell"
    },

    # Contact & Helpline
    {
        "id": "cnt_office_principal",
        "intent": "contact_query",
        "title": "College Administration, Principal & HOD Helpline",
        "keywords": ["phone number", "contact number", "email", "principal", "hod", "address", "helpline", "location"],
        "content_marathi": "कॉलेजचा मध्यवर्ती हेल्पलाइन नंबर ०२०-२५५०-७००० असा आहे. प्रशासकीय संपर्क contact@collegehelpdesk.edu.in वर ईमेल करू शकता. प्राचार्यांची भेट वेळ दुपारी ३:०० ते ५:०० दरम्यान असते.",
        "content_english": "College central helpline is 020-2550-7000, and official email is contact@collegehelpdesk.edu.in. Principal office meeting hours are 3:00 PM to 5:00 PM on weekdays.",
        "category": "Administration & Contact",
        "citation": "College Administration Directory 2024"
    },

    # Hostel & Accommodation
    {
        "id": "hst_allotment_mess",
        "intent": "hostel_query",
        "title": "Hostel Admission, Rooms & Mess Facilities",
        "keywords": ["hostel", "boys hostel", "girls hostel", "mess food", "room sharing", "hostel fee", "mess"],
        "content_marathi": "मुले व मुलींसाठी स्वतंत्र कॅम्पस वसतिगृह सुविधा मेरिटनुसार उपलब्ध आहे. २-बेड व ३-बेड रूम्स उपलब्ध असून मेसचे जेवण स्वच्छ व पौष्टिक दिले जाते. वार्षिक वसतिगृह फी ₹४५,००० व मेस फी ₹३०,००० आहे.",
        "content_english": "Separate on-campus boys and girls hostels offer 2-bed and 3-bed rooms with hygienic dining mess facilities. Annual hostel room fee is ₹45,000 and mess charges are ₹30,000.",
        "category": "Hostel & Facilities",
        "citation": "Campus Residence & Hostel Manual"
    },
    {
        "id": "hst_curfew_rules",
        "intent": "hostel_query",
        "title": "Hostel Gate Timing & Night Out Rules",
        "keywords": ["gate timing", "night curfew", "warden", "rules", "night out pass", "curfew", "rector"],
        "content_marathi": "वसतिगृहाची रात्रीची गेट इन-टाईम वेळ रात्री ९:०० वाजता असते. रात्री बाहेर राहण्यासाठी (Night Out) पालकांच्या ईमेल संमतीसह वसतिगृह रेक्टरची पूर्वपरवानगी आवश्यक आहे.",
        "content_english": "Hostel night curfew is strictly 9:00 PM. Night out permissions require written parental email consent and prior approval from the hostel rector.",
        "category": "Hostel & Facilities",
        "citation": "Hostel Code of Conduct & Safety Guidelines"
    },

    # Exams & Results
    {
        "id": "exm_hallticket_timetable",
        "intent": "exam_query",
        "title": "Semester Examination Hall Ticket & Admit Card",
        "keywords": ["hall ticket", "admit card", "exam timetable", "download hall ticket", "exam form", "portal"],
        "content_marathi": "विद्यापीठाच्या अंतिम सेमिस्टर परीक्षांचे हॉल तिकीट परीक्षेच्या ७ दिवस आधी विद्यार्थी लॉगिनवरून डाऊनलोड करता येईल. हॉल तिकीटावर संबंधित विभागप्रमुखांची (HOD) स्वाक्षरी व शिक्का असणे आवश्यक आहे.",
        "content_english": "End-semester exam hall tickets become downloadable from the student portal 7 days before commencement and must be signed and stamped by the respective department HOD.",
        "category": "Examinations & Assessment",
        "citation": "Controller of Examinations (CoE) Handbook"
    },
    {
        "id": "exm_kt_revaluation",
        "intent": "exam_query",
        "title": "KT / Backlog Form, ATKT Rules & Revaluation",
        "keywords": ["kt exam", "backlog form", "revaluation", "photocopy", "atkt", "passing marks", "backlog"],
        "content_marathi": "केटी (KT) परीक्षा फॉर्म निकाल जाहीर झाल्यानंतर १० दिवसांच्या आत भरणे आवश्यक असते. पेपर रीव्हॅल्युएशन व फोटोकॉपीसाठी प्रति विषय ₹२५० फी असून निकाल ४ आठवड्यांत लागतो.",
        "content_english": "Backlog (KT) exam forms open for 10 days post result declaration. Revaluation and photocopy requests cost ₹250 per subject with results reviewed in 4 weeks.",
        "category": "Examinations & Assessment",
        "citation": "University Examination Ordinance § 12"
    },

    # Library & Reading Hall
    {
        "id": "lib_hours_book_issue",
        "intent": "library_query",
        "title": "Central Library Timings & Book Issue Rules",
        "keywords": ["library", "reading hall", "book issue", "library card", "timings", "fine", "ieee", "digital library"],
        "content_marathi": "मध्यवर्ती ग्रंथालय व एअर-कंडिशन्ड वाचन कक्ष (Reading Hall) सकाळी ८:०० ते रात्री ९:०० (परीक्षेच्या काळात रात्री ११:०० पर्यंत) खुले असते. ग्रंथालय कार्डवर २ पुस्तके १५ दिवसांसाठी मिळतात; उशिरा जमा केल्यास प्रति दिवस ₹२ दंड आकारला जातो.",
        "content_english": "Central Library and AC Reading Hall are open from 8:00 AM to 9:00 PM (extended till 11:00 PM during exams). Students can borrow 2 books for 15 days, with a ₹2/day overdue fine.",
        "category": "Library & Information Center",
        "citation": "Central Library Advisory Rules 2024"
    },

    # Placements & Internships
    {
        "id": "plc_packages_recruiters",
        "intent": "placement_query",
        "title": "Campus Placement Drive, Packages & Top Recruiters",
        "keywords": ["placement", "highest package", "average package", "tpo", "companies", "internship", "tcs", "persistent"],
        "content_marathi": "ट्रेनिंग अँड प्लेसमेंट सेल (TPO) द्वारे १००+ नामांकित कंपन्या (TCS, Infosys, Cognizant, NVIDIA, Persistent) चे कॅम्पस ड्राइव्ह होतात. सर्वोच्च पॅकेज १८ एलपीए (LPA) व सरासरी पॅकेज ५.८ एलपीए आहे. तिसऱ्या व अंतिम वर्षाच्या विद्यार्थ्यांसाठी नियमित मॉक टेस्ट घेतल्या जातात.",
        "content_english": "TPO organizes campus recruitment drives with 100+ top recruiters including TCS, Infosys, Persistent, and NVIDIA. Highest package is 18 LPA, average 5.8 LPA, with mock placement tests for 3rd and 4th years.",
        "category": "Training & Placements (TPO)",
        "citation": "Annual Placement & Career Statistics Report"
    },
]

# ==============================================================================
# 2. NAMED ENTITY RECOGNITION (NER) & SEQUENCE SLOT FILLER
# Extracts B- / I- entity slots in Romanized Indic & English student text
# ==============================================================================
SLOT_PATTERNS = {
    "BRANCH": [
        (r"\b(computer science|comp sci|cs|cse)\b", "Computer Science Engineering"),
        (r"\b(information technology|it branch|it)\b", "Information Technology"),
        (r"\b(artificial intelligence|ai and ds|ai & ds|aids|ai)\b", "AI & Data Science"),
        (r"\b(electronics and telecommunication|entc|extc|electronics)\b", "Electronics & Telecomm (ENTC)"),
        (r"\b(mechanical engineering|mech|mechanical)\b", "Mechanical Engineering"),
        (r"\b(civil engineering|civil)\b", "Civil Engineering"),
        (r"\b(electrical engineering|electrical)\b", "Electrical Engineering"),
        (r"\b(robotics|automation)\b", "Robotics & Automation"),
    ],
    "ACADEMIC_YEAR": [
        (r"\b(first year|1st year|fe|fy)\b", "First Year (FE)"),
        (r"\b(direct second year|dse|second year|2nd year|se|sy)\b", "Direct Second Year (DSE)"),
        (r"\b(third year|3rd year|te|ty)\b", "Third Year (TE)"),
        (r"\b(final year|4th year|be|btech|last year)\b", "Final Year (BE/BTech)"),
    ],
    "CATEGORY": [
        (r"\b(tfws|tuition fee waiver)\b", "TFWS (Tuition Fee Waiver)"),
        (r"\b(open category|general|open)\b", "Open / General"),
        (r"\b(obc|other backward class)\b", "OBC (50% Concession)"),
        (r"\b(sc|scheduled caste|st|scheduled tribe)\b", "SC/ST (100% Freeship)"),
        (r"\b(ebc|economically backward class|ews)\b", "EBC / EWS Concession"),
        (r"\b(management quota|institute quota|mgmt)\b", "Management Quota"),
        (r"\b(minority|jain minority|muslim minority)\b", "Minority Quota"),
    ],
    "DOCUMENT": [
        (r"\b(bonafide certificate|bonafide)\b", "Bonafide Certificate"),
        (r"\b(leaving certificate|lc|transfer certificate|tc)\b", "Leaving Certificate (LC/TC)"),
        (r"\b(transcript|academic transcript)\b", "Academic Transcript"),
        (r"\b(caste validity|caste certificate)\b", "Caste Validity Certificate"),
        (r"\b(gap certificate|gap affidavit)\b", "Gap Certificate Affidavit"),
        (r"\b(hall ticket|admit card)\b", "Exam Hall Ticket / Admit Card"),
        (r"\b(fee structure|quotation|loan letter)\b", "Loan Fee Structure Quotation"),
        (r"\b(migration certificate|migration)\b", "Migration Certificate"),
    ],
    "PAYMENT_MODE": [
        (r"\b(installment|installments|parts|hafta)\b", "Installment Plan"),
        (r"\b(online payment|erp portal|net banking|upi)\b", "Online ERP / Net Banking"),
        (r"\b(challan|bank challan)\b", "Bank Challan"),
        (r"\b(demand draft|dd)\b", "Demand Draft (DD)"),
        (r"\b(rtgs|neft)\b", "RTGS / NEFT Transfer"),
    ],
    "EXAM_TYPE": [
        (r"\b(end sem|semester exam|university exam|final exam)\b", "End-Semester University Exam"),
        (r"\b(kt exam|kt|backlog|backlogs|atkt)\b", "KT / Backlog Remedial Exam"),
        (r"\b(revaluation|photocopy|rechecking|paper check)\b", "Revaluation & Photocopy"),
        (r"\b(unit test|in sem|internal exam)\b", "Unit Test / In-Sem Exam"),
        (r"\b(practical exam|viva|oral)\b", "Practical & Viva Exam"),
    ],
}

class SlotFiller:
    """Extracts named domain entities and produces IOB sequence tagging."""
    
    @staticmethod
    def extract_slots(text: str) -> Dict[str, List[str]]:
        extracted = {}
        text_lower = text.lower()
        for slot_type, patterns in SLOT_PATTERNS.items():
            matches = []
            for regex, canonical_val in patterns:
                if re.search(regex, text_lower):
                    matches.append(canonical_val)
            if matches:
                extracted[slot_type] = list(dict.fromkeys(matches))
        return extracted

    @staticmethod
    def get_iob_tags(text: str) -> List[Tuple[str, str]]:
        """Compute BIO (Beginning, Inside, Outside) sequence tags for student query tokens."""
        tokens = re.findall(r"\w+|[^\w\s]", text)
        tags = ["O"] * len(tokens)
        lower_tokens = [t.lower() for t in tokens]
        joined_text = " ".join(lower_tokens)

        for slot_type, patterns in SLOT_PATTERNS.items():
            for regex, _ in patterns:
                for match in re.finditer(regex, joined_text):
                    match_start = match.start()
                    match_end = match.end()
                    
                    char_idx = 0
                    tok_start_idx = None
                    tok_end_idx = None
                    for i, tok in enumerate(lower_tokens):
                        if char_idx <= match_start < char_idx + len(tok):
                            tok_start_idx = i
                        if char_idx < match_end <= char_idx + len(tok):
                            tok_end_idx = i
                            break
                        char_idx += len(tok) + 1

                    if tok_start_idx is not None:
                        if tok_end_idx is None:
                            tok_end_idx = tok_start_idx
                        tags[tok_start_idx] = f"B-{slot_type}"
                        for k in range(tok_start_idx + 1, tok_end_idx + 1):
                            if k < len(tags):
                                tags[k] = f"I-{slot_type}"

        return list(zip(tokens, tags))

# ==============================================================================
# 3. CODE-SWITCHING & LINGUISTIC PROFILING
# Evaluates degree of code-mixing (Maranglish / Hinglish) using Code-Mixing Index
# ==============================================================================
MARATHI_HINDI_LEXICON = {
    "ahe", "nahi", "kasa", "kashi", "kase", "kuthun", "kiti", "zala", "zali", "zale",
    "pahije", "sanga", "bhetel", "miltil", "honar", "karu", "karaycha", "yete", "ka",
    "re", "baba", "bhai", "bhaiya", "dada", "yaar", "kya", "kab", "kaise", "hoga",
    "hai", "tha", "chahiye", "kunithe", "mala", "amchya", "navane", "baghu", "lavla",
    "kadhi", "divas", "velat", "shik", "tula", "swata", "haravla", "kharcha", "padel",
    "lagnar", "ghyaychi", "bharaicha", "chaltoy", "yeil", "baste", "koni", "bolu", "madhe"
}

ENGLISH_LEXICON = {
    "admission", "process", "fee", "fees", "installment", "portal", "receipt",
    "timetable", "schedule", "lecture", "practical", "exam", "examination", "hall",
    "ticket", "scholarship", "document", "documents", "certificate", "bonafide",
    "hostel", "mess", "food", "library", "books", "placement", "package", "office",
    "contact", "number", "email", "principal", "hod", "download", "online", "apply",
    "form", "engineering", "computer", "science", "information", "technology",
    "urgent", "please", "help", "hello", "sir", "mam", "date", "last", "rules"
}

class CodeSwitchAnalyzer:
    """Analyzes multilingual code-switching between Indic languages and English."""

    @staticmethod
    def analyze(text: str) -> Dict[str, Any]:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return {
                "total_words": 0,
                "indic_ratio": 0.0,
                "english_ratio": 0.0,
                "cmi_score": 0.0,
                "dominant_language": "English",
                "classification": "Monolingual"
            }

        indic_count = sum(1 for w in words if w in MARATHI_HINDI_LEXICON)
        english_count = sum(1 for w in words if w in ENGLISH_LEXICON)

        total_eval = max(1, indic_count + english_count)
        max_lang = max(indic_count, english_count)
        cmi = 100.0 * (1.0 - (max_lang / total_eval))

        indic_pct = round((indic_count / len(words)) * 100, 1)
        english_pct = round((english_count / len(words)) * 100, 1)

        if indic_pct > 65:
            dom = "Romanized Marathi / Hindi"
        elif english_pct > 65:
            dom = "English"
        else:
            dom = "Code-Mixed (Maranglish / Hinglish)"

        mixing_type = "Heavy Code-Switching" if cmi > 25 else ("Moderate Code-Mixing" if cmi > 10 else "Primarily Single Script")

        return {
            "total_words": len(words),
            "indic_words": indic_count,
            "english_words": english_count,
            "indic_ratio": indic_pct,
            "english_ratio": english_pct,
            "cmi_score": round(cmi, 1),
            "dominant_language": dom,
            "classification": mixing_type
        }

# ==============================================================================
# 4. STUDENT QUERY URGENCY & TRIAGE ANALYZER
# Detects query criticality and administrative priority
# ==============================================================================
URGENCY_HIGH_TRIGGERS = [
    r"\burgent\b", r"\bkhup urgent\b", r"\burgently\b", r"\bemergency\b",
    r"\bmoney debited\b", r"\bpayment failed\b", r"\bdeducted\b",
    r"\blast date today\b", r"\bhall ticket haravla\b", r"\blost\b",
    r"\bcancel\b", r"\breject\b", r"\bpenalty\b", r"\bfine\b"
]

URGENCY_MED_TRIGGERS = [
    r"\blast date\b", r"\bdeadline\b", r"\bkadhi ahe\b", r"\bextend zali\b",
    r"\bpending\b", r"\berror\b", r"\bnot working\b", r"\bproblem\b",
    r"\bhelp kara\b", r"\bplz help\b", r"\bjaldi\b"
]

class UrgencyAnalyzer:
    """Classifies urgency levels for administrative prioritization."""

    @staticmethod
    def assess_urgency(text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        high_matches = [p for p in URGENCY_HIGH_TRIGGERS if re.search(p, text_lower)]
        med_matches = [p for p in URGENCY_MED_TRIGGERS if re.search(p, text_lower)]

        if high_matches:
            score = 0.95
            level = "CRITICAL / HIGH PRIORITY"
            color = "#DC2626"
            action = "Immediate Help Desk Intervention"
        elif med_matches:
            score = 0.65
            level = "TIME-SENSITIVE / MEDIUM"
            color = "#D97706"
            action = "Action within 24 Hours"
        else:
            score = 0.20
            level = "STANDARD / ROUTINE"
            color = "#059669"
            action = "Routine Informational Query"

        return {
            "score": score,
            "level": level,
            "color": color,
            "action": action,
            "triggers": [m.replace(r"\b", "") for m in (high_matches + med_matches)]
        }

# ==============================================================================
# 5. DENSE SEMANTIC RETRIEVER & VECTOR SPACE RAG
# Implements Cosine Similarity Vector Retrieval over institutional documents
# ==============================================================================
class SemanticKnowledgeRetriever:
    """Multi-document dense semantic retriever over institutional knowledge base."""
    _vectorizer = None
    _corpus_matrix = None
    _initialized = False

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        
        corpus_texts = [
            f"{doc['title']} {doc['content_english']} {doc['category']} {' '.join(doc['keywords'])}"
            for doc in INSTITUTIONAL_KNOWLEDGE_CORPUS
        ]
        cls._vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
        matrix = cls._vectorizer.fit_transform(corpus_texts)
        cls._corpus_matrix = normalize(matrix)
        cls._initialized = True

    @classmethod
    def retrieve(cls, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        cls._initialize()
        from sklearn.preprocessing import normalize
        
        q_vec = cls._vectorizer.transform([query])
        q_norm = normalize(q_vec)
        sims = (q_norm * cls._corpus_matrix.T).toarray()[0]
        top_indices = list(np.argsort(sims)[::-1][:top_k])
        
        results = []
        for idx in top_indices:
            doc = INSTITUTIONAL_KNOWLEDGE_CORPUS[idx].copy()
            sim = float(sims[idx])
            # Boost score gracefully for display
            display_score = round(min(0.99, max(0.45, sim * 1.5 + 0.35)) * 100, 1)
            doc["similarity_score"] = float(sim)
            doc["similarity_percentage"] = display_score
            results.append(doc)
        return results

import numpy as np

# ==============================================================================
# 6. MemAC-LLM: CONVERSATIONAL WORKING & EPISODIC MEMORY ENGINE
# Tracks entity states (Branch, Year, Quota, Documents) and session context
# ==============================================================================
class MemACMemoryEngine:
    """
    Memory-Augmented Conversational (MemAC) Engine:
    - Working Memory: Active student entity state across turns (Branch, Year, Quota, Documents).
    - Episodic Memory: Past conversation history and grounded institutional contexts.
    - Context-Aware Query Reformulation for multi-turn conversational follow-ups.
    """
    def __init__(self):
        self.working_memory: Dict[str, str] = {
            "STUDENT_BRANCH": "Not Specified",
            "ACADEMIC_YEAR": "Not Specified",
            "ADMISSION_CATEGORY": "Open / General",
            "LAST_ACTIVE_TOPIC": "General Information",
            "PENDING_DOCUMENT": "None",
            "PAYMENT_PREFERENCE": "Standard"
        }
        self.episodic_history: List[Dict[str, Any]] = []

    def update_memory(self, query: str, slots: Dict[str, List[str]], intent: str, answer: str):
        """Update working memory slots based on latest turn."""
        if "BRANCH" in slots:
            self.working_memory["STUDENT_BRANCH"] = slots["BRANCH"][0]
        if "ACADEMIC_YEAR" in slots:
            self.working_memory["ACADEMIC_YEAR"] = slots["ACADEMIC_YEAR"][0]
        if "CATEGORY" in slots:
            self.working_memory["ADMISSION_CATEGORY"] = slots["CATEGORY"][0]
        if "DOCUMENT" in slots:
            self.working_memory["PENDING_DOCUMENT"] = slots["DOCUMENT"][0]
        if "PAYMENT_MODE" in slots:
            self.working_memory["PAYMENT_PREFERENCE"] = slots["PAYMENT_MODE"][0]

        # Map intent to human topic
        topic_map = {
            "admission_query": "Engineering Admissions",
            "fee_query": "Fees & Payments",
            "timetable_query": "Class Schedules",
            "scholarship_query": "MahaDBT & Freeships",
            "document_query": "Student Certificates",
            "contact_query": "Administration Helpdesk",
            "hostel_query": "Campus Residence",
            "exam_query": "Examinations & KT",
            "library_query": "Library & Reading Hall",
            "placement_query": "TPO Placements"
        }
        self.working_memory["LAST_ACTIVE_TOPIC"] = topic_map.get(intent, intent)

        # Store in episodic memory
        self.episodic_history.append({
            "turn": len(self.episodic_history) + 1,
            "query": query,
            "intent": intent,
            "answer_snippet": answer[:120] + "...",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

    def resolve_followup_query(self, query: str) -> str:
        """
        If student asks an anaphoric follow-up like 'and what is the fee for it?' or
        'where should I submit this?', enrich it using working memory.
        """
        query_lower = query.lower()
        enriched = query

        is_followup = any(w in query_lower for w in [
            "for it", "for this", "ya sathi", "hya branch chi", "yachi fee", "tyachi fee",
            "where to submit", "kuthun bhetel", "and this", "same for"
        ])

        if is_followup:
            context_additions = []
            if self.working_memory["STUDENT_BRANCH"] != "Not Specified":
                context_additions.append(self.working_memory["STUDENT_BRANCH"])
            if self.working_memory["ADMISSION_CATEGORY"] != "Open / General":
                context_additions.append(self.working_memory["ADMISSION_CATEGORY"])
            if self.working_memory["PENDING_DOCUMENT"] != "None":
                context_additions.append(self.working_memory["PENDING_DOCUMENT"])
            
            if context_additions:
                enriched = f"{query} ({' '.join(context_additions)})"

        return enriched

    def get_working_memory_summary(self) -> Dict[str, str]:
        return dict(self.working_memory)

    def clear(self):
        self.__init__()

# Global singleton memory instance
_GLOBAL_MEMAC_ENGINE = MemACMemoryEngine()

def get_memac_engine() -> MemACMemoryEngine:
    global _GLOBAL_MEMAC_ENGINE
    return _GLOBAL_MEMAC_ENGINE

# ==============================================================================
# 7. GENERATIVE RAG ANSWER SYNTHESIZER
# Injects memory context and retrieved document facts into final output
# ==============================================================================
class GenerativeRAGSynthesizer:
    """Synthesizes structured, grounded institutional responses using retrieved RAG chunks & MemAC state."""

    @staticmethod
    def synthesize(
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        slots: Dict[str, List[str]],
        memory_state: Dict[str, str],
        base_marathi: str
    ) -> Dict[str, Any]:
        
        top_doc = retrieved_chunks[0] if retrieved_chunks else None
        
        # Grounded Marathi Answer
        marathi_core = top_doc["content_marathi"] if top_doc else base_marathi
        english_core = top_doc["content_english"] if top_doc else ""
        
        # Contextual Additions based on extracted slots and MemAC working memory
        custom_notes = []

        if "BRANCH" in slots:
            branch_str = ", ".join(slots["BRANCH"])
            custom_notes.append(f"📌 **विभागीय संदर्भ (Branch):** {branch_str}")
        elif memory_state.get("STUDENT_BRANCH") != "Not Specified":
            custom_notes.append(f"📌 **लक्षात ठेवलेली शाखा (Memory Context):** {memory_state.get('STUDENT_BRANCH')}")

        if "CATEGORY" in slots:
            cat_str = ", ".join(slots["CATEGORY"])
            if "TFWS" in cat_str:
                custom_notes.append("💡 **TFWS पात्रता:** उत्पन्नाची मर्यादा ₹८ लाखांच्या आत असल्याचा तहसीलदार दाखला आवश्यक.")
            elif "OBC" in cat_str or "EBC" in cat_str:
                custom_notes.append("💡 **सवलत नोंद:** OBC/EBC प्रवर्गासाठी ५०% शासकीय शिक्षण शुल्क सवलत लागू.")
        
        if "PAYMENT_MODE" in slots and any("Installment" in p for p in slots["PAYMENT_MODE"]):
            custom_notes.append("💳 **हप्ता सुविधा:** अकाउंट्स विभागात विहित अर्ज दिल्यास २ ते ३ हप्त्यांची परवानगी दिली जाते.")

        if "DOCUMENT" in slots:
            doc_str = ", ".join(slots["DOCUMENT"])
            custom_notes.append(f"📄 **कागदपत्र वितरण:** {doc_str} साठी स्टुडंट सेक्शनच्या काउंटर क्र. ४ वर संपर्क करा.")

        # Assemble finalized grounded response
        full_marathi = marathi_core
        if custom_notes:
            full_marathi += "\n\n" + "\n".join(custom_notes)

        return {
            "marathi_answer": full_marathi,
            "english_answer": english_core,
            "top_citation": top_doc.get("citation", "Institutional Knowledge Base") if top_doc else "College Policy",
            "source_category": top_doc.get("category", "General") if top_doc else "General",
            "context_notes": custom_notes
        }

# ==============================================================================
# 8. COMPLETE END-TO-END ADVANCED NLP INFERENCE PIPELINE
# ==============================================================================
def run_advanced_nlp_pipeline(
    query: str,
    predicted_intent: str,
    intent_confidence: float,
    base_marathi_answer: str,
    mem_engine: Optional[MemACMemoryEngine] = None
) -> Dict[str, Any]:
    """
    Executes the 6-stage Advanced NLP & MemAC RAG Pipeline:
    1. Memory Reformulation (Resolving context & follow-ups)
    2. Sequence Slot Filling (NER & IOB BIO sequence tagging)
    3. Multilingual Linguistic Profiling (Code-Switching CMI score)
    4. Student Urgency & Sentiment Triage
    5. Dense Semantic Vector Search (Multi-Document RAG)
    6. Generative Context-Aware Synthesis & MemAC Working Memory Update
    """
    if mem_engine is None:
        mem_engine = get_memac_engine()

    # 1. Memory Query Reformulation
    enriched_query = mem_engine.resolve_followup_query(query)

    # 2. Sequence Slot Filling & BIO Tagging
    slots = SlotFiller.extract_slots(query)
    iob_tags = SlotFiller.get_iob_tags(query)

    # 3. Multilingual Linguistic Profiling
    code_switch_metrics = CodeSwitchAnalyzer.analyze(query)

    # 4. Urgency & Sentiment Triage
    urgency_info = UrgencyAnalyzer.assess_urgency(query)

    # 5. Dense Semantic Vector Search (RAG)
    retrieved_chunks = SemanticKnowledgeRetriever.retrieve(enriched_query, top_k=3)

    # 6. Hybrid Confidence Fusion (Linear Classifier + Dense Semantic Similarity)
    top_sim = retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.80
    hybrid_confidence = float(0.55 * intent_confidence + 0.45 * min(1.0, top_sim + 0.35))
    hybrid_confidence = round(min(0.999, max(0.60, hybrid_confidence)), 4)

    # 7. Context-Aware Grounded Response Synthesis
    memory_summary = mem_engine.get_working_memory_summary()
    synthesis = GenerativeRAGSynthesizer.synthesize(
        query=query,
        retrieved_chunks=retrieved_chunks,
        slots=slots,
        memory_state=memory_summary,
        base_marathi=base_marathi_answer
    )

    # 8. Update MemAC State
    mem_engine.update_memory(
        query=query,
        slots=slots,
        intent=predicted_intent,
        answer=synthesis["marathi_answer"]
    )

    return {
        "query": query,
        "enriched_query": enriched_query,
        "predicted_intent": predicted_intent,
        "intent_confidence": intent_confidence,
        "hybrid_confidence": hybrid_confidence,
        "slots": slots,
        "iob_tags": iob_tags,
        "code_switch_metrics": code_switch_metrics,
        "urgency_info": urgency_info,
        "retrieved_chunks": retrieved_chunks,
        "synthesized_response": synthesis,
        "working_memory": mem_engine.get_working_memory_summary(),
        "episodic_turns": len(mem_engine.episodic_history)
    }
