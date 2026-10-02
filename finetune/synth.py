"""Generate labelled training texts.

The point is not volume, it is contrast. Every template slot is one of:

    {name}    a PII value      -> its span is labelled positive
    {~phone}  a PII-SHAPED value in a non-personal role (a reference number that happens to
              look like a phone number) -> deliberately NOT labelled
    {city}    an ordinary decoy -> not labelled

`{~slot}` is what teaches the model the rule the whole system rests on:

    PII-looking value + personal context     -> PII
    PII-looking value + non-personal context -> NOT PII

Without those, the model can reach high accuracy by learning formats alone, which is exactly
the failure mode this data is meant to prevent. Texts also vary in phrasing, position, casing,
punctuation, language and tidiness, so the model cannot lean on a fixed sentence shape.
"""

import random
import re
import string

FIRST = ("Rahul Priya Anil Sunita Meera Karthik Arjun Neha Amit Ramesh Kavya Rohit Deepa Suresh Lakshmi Vikram "
         "Pooja Imran Fatima Sanjay Harpreet Ritika Varun Nisha Gopal Anjali Manoj Aisha Kiran Siddharth Rekha Anand "
         "Daniel Emily Sarah Michael Olivia James Grace Kevin Rachel Samuel Liam Emma Lucas Sofia Klaus Yuki Ahmed "
         "Wei Carlos Jean Claire Mohammed Hannah Tom Ben Maria John Lisa Robert Jennifer David Chloe Omar Elena "
         "Farhan Geeta Kunal Pradeep Radha Shalini Venkatesh Arvind Nikhil Aarti Devika Ishaan Tanvi Yash Zara").split()
LAST = ("Sharma Venkataraman Kapoor Devi Iyer Mehta Reddy Kumar Patel Nair Verma Menon Prasad Qureshi Sheikh Kulkarni "
        "Singh Sen Malhotra Agarwal Rao Gupta Tiwari Khan Joshi Bhat Das Pillai Krishnan Swamy Choudhary Shetty "
        "Morgan Carter Connor Brown Anderson Kim Green Jackson Wilson Watson Martin Hernandez Becker Tanaka Hassan "
        "Chen Silva Dubois Ali Lee Hardy Thompson Rodriguez Lopez Garcia Wright Schmidt Rossi Novak Yadav Bose").split()
CITIES = ("Bangalore Mumbai Delhi Chennai Kolkata Hyderabad Pune Jaipur Lucknow Kochi Mangalore Varanasi Agra Patna "
          "Coimbatore Gurgaon London Paris Berlin Tokyo Sydney Dubai Madrid Austin Chicago Seattle Toronto Singapore").split()
STATES = "Karnataka Maharashtra Kerala Telangana Rajasthan Texas California Ontario Bavaria Gujarat Punjab".split()
COMPANIES = ("Infosys Wipro Microsoft Google Amazon Flipkart Swiggy Zomato Apple Oracle Salesforce Accenture Deloitte "
             "Tata Reliance Paytm Razorpay Stripe Adobe Netflix Uber Airbnb Samsung Sony Toyota Honda Siemens").split()
PRODUCTS = "Azure Kubernetes Terraform PostgreSQL Redis Kafka Jenkins Jira Slack Figma iPhone Pixel Nexon Swift Grafana".split()
STREETS = ("MG Road|Brigade Road|Park Street|Nehru Nagar|Baker Street|Elm Street|Lake View Road|Residency Road|"
           "Civil Lines|Linking Road|Oak Avenue|Maple Lane|Anna Salai|Church Street").split("|")
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
TITLES = ("Senior Analyst|Product Manager|Software Engineer|Team Lead|Account Manager|Data Scientist|"
          "Regional Head|Support Engineer|Finance Controller").split("|")
LANDMARKS = "Taj Mahal|Eiffel Tower|World Trade Center|Tidel Park|Victoria Memorial|Charminar|India Gate".split("|")


def d(n):
    return "".join(random.choice(string.digits) for _ in range(n))


def up(n):
    return "".join(random.choice(string.ascii_uppercase) for _ in range(n))


def name():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def phone():
    return random.choice([
        f"+91 9{d(4)} {d(5)}", f"9{d(9)}", f"+91-9{d(4)}-{d(5)}", f"({d(3)}) 555-{d(4)}", f"+1 {d(3)}-555-{d(4)}",
        f"+44 7{d(3)} {d(6)}", f"0{d(4)} {d(6)}", f"+971 50 {d(3)} {d(4)}", f"+33 6 {d(2)} {d(2)} {d(2)} {d(2)}",
        f"+49 30 {d(4)} {d(4)}", f"+65 9{d(3)} {d(4)}", f"080 {d(4)} {d(4)}", f"+86 138 {d(4)} {d(4)}",
    ])


def email(n=None):
    n = (n or name()).lower().split()
    user = random.choice([f"{n[0]}.{n[1]}", f"{n[0]}{d(2)}", f"{n[0][0]}{n[1]}", f"{n[0]}_{n[1][:3]}", n[0]])
    dom = random.choice(["gmail.com", "yahoo.co.in", "outlook.com", "example.com", "corp.example",
                         "proton.me", "mail.fr", "example.org", "company.example", "hospital.example"])
    return f"{user}@{dom}"


def date():
    return random.choice([f"{random.randint(1, 28)} {random.choice(MONTHS)} {random.randint(1950, 2005)}",
                          f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1950, 2005)}",
                          f"{random.randint(1950, 2005)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"])


HI_FIRST = "कविता अमित सुनीता राहुल प्रिया अनिल मीरा संजय नेहा विकास पूजा रमेश".split()
HI_LAST = "शर्मा वर्मा गुप्ता कुमार सिंह यादव जोशी नायर मेहता पटेल".split()
TA_NAME = "முருகன்|கவிதா|ராஜேஷ்|பிரியா|அன்பு செல்வன்|மீனா".split("|")

PII = {
    "hi_name": lambda: f"{random.choice(HI_FIRST)} {random.choice(HI_LAST)}",
    "ta_name": lambda: random.choice(TA_NAME),
    "name": name,
    "first": lambda: random.choice(FIRST),
    "lower_first": lambda: random.choice(FIRST).lower(),
    "lower_name": lambda: name().lower(),
    "upper_name": lambda: name().upper(),
    "mixed_name": lambda: f"{random.choice(FIRST)} {random.choice(LAST).upper()}",
    "phone": phone,
    "email": email,
    "dob": date,
    "aadhaar": lambda: f"{random.randint(2, 9)}{d(3)} {d(4)} {d(4)}",
    "pan": lambda: f"{up(5)}{d(4)}{up(1)}",
    "ssn": lambda: f"{d(3)}-{d(2)}-{d(4)}",
    "passport": lambda: f"{up(1)}{d(random.choice([7, 8]))}",
    "card": lambda: f"{random.choice('3456')}{d(3)} {d(4)} {d(4)} {d(4)}",
    "last4": lambda: d(4),
    "account": lambda: d(random.choice([10, 11, 12, 14])),
    "upi": lambda: f"{random.choice(FIRST).lower()}{random.choice(['', '.', '_'])}{d(2)}@{random.choice(['upi', 'okaxis', 'ybl', 'paytm'])}",
    "cvv": lambda: d(3),
    "otp": lambda: d(6),
    "password": lambda: random.choice(["Summer", "Blue#Sky", "Temp!Pass", "Qwerty!", "Hunter2!", "MySecret", "P@ssw0rd"]) + d(random.choice([2, 3, 4])),
    "username": lambda: random.choice([f"{random.choice(FIRST).lower()}_{d(2)}", f"{random.choice(FIRST).lower()}.{random.choice(LAST)[0].lower()}", f"admin_{random.choice(FIRST).lower()}"]),
    "apikey": lambda: random.choice([f"sk-live-{d(4)}{up(4).lower()}{d(6)}", f"ghp_{up(6)}{d(6)}{up(4).lower()}", f"AKIA{up(12)}{d(4)}"]),
    "ip": lambda: f"{random.randint(11, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
    "mac": lambda: ":".join(f"{random.randint(0, 255):02X}" for _ in range(6)),
    "mrn": lambda: d(random.choice([6, 7, 8])),
    "member_id": lambda: f"MBR-{d(7)}",
    "emp_id": lambda: f"EMP-{d(random.choice([5, 6]))}",
    "policy": lambda: f"POL-{d(random.choice([7, 8]))}",
    "vehicle": lambda: f"{up(2)}{d(2)}{up(2)}{d(4)}",
    "address": lambda: f"{random.randint(1, 999)} {random.choice(STREETS)}",
    "flat": lambda: f"Flat {random.randint(1, 20)}{random.choice('ABCD')}, {random.choice(['Green Heights', 'Lotus Towers', 'Sunshine Apartments', 'Palm Residency'])}",
    "pincode": lambda: f"{random.randint(1, 8)}{d(5)}",
    "zip": lambda: d(5),
}
DECOY = {
    "city": lambda: random.choice(CITIES),
    "state": lambda: random.choice(STATES),
    "company": lambda: random.choice(COMPANIES),
    "product": lambda: random.choice(PRODUCTS),
    "day": lambda: random.choice(DAYS),
    "title": lambda: random.choice(TITLES),
    "landmark": lambda: random.choice(LANDMARKS),
    "ticket": lambda: f"{random.choice(['JIRA', 'SUP', 'DEV', 'OPS', 'TKT'])}-{d(random.choice([4, 5]))}",
    "order": lambda: f"{random.choice(['ORD', 'INV', 'SKU', 'PO'])}-{d(random.choice([4, 5, 6]))}",
    "version": lambda: f"{random.randint(1, 9)}.{random.randint(0, 20)}.{random.randint(0, 9)}",
    "amount": lambda: random.choice([f"₹{random.randint(1, 99)},{d(3)}", f"${random.randint(1, 999)}.{d(2)}", f"${random.randint(1, 9)}.{random.randint(1, 9)}M", f"{random.randint(1, 99)}%"]),
    "time": lambda: random.choice([f"{random.randint(1, 12)}:{random.choice(['00', '15', '30', '45'])} {random.choice(['AM', 'PM'])}", f"{d(2)}:{d(2)} UTC", f"{random.randint(1, 11)}pm"]),
    "count": lambda: random.choice([f"{random.randint(2, 999)}", f"{random.randint(1, 99)},{d(3)}"]),
    "room": lambda: f"Room {random.randint(100, 1500)}",
    "build": lambda: f"Build {d(4)}",
    "ifsc": lambda: f"{up(4)}0{d(6)}",
    "year": lambda: str(random.randint(1900, 2030)),
    "month_date": lambda: f"{random.randint(1, 28)} {random.choice(MONTHS)} 20{random.randint(24, 27)}",
    "hsn": lambda: d(4),
    "sprint": lambda: f"Sprint {random.randint(1, 99)}",
    "commit": lambda: "".join(random.choice("0123456789abcdef") for _ in range(7)),
    "port": lambda: str(random.randint(1000, 9999)),
}

# ------------------------------------------------------------------ templates
# {slot} = PII, {~slot} = a PII-shaped value in a non-personal role, other = decoy.

NAME_PHRASINGS = [
    "My name is {name}.", "This is {name}.", "I am {name}.", "Please contact {name}.",
    "I spoke with {name} yesterday.", "Can you ask {name} to call me?", "The customer is {name}.",
    "The account belongs to {name}.", "{name} contacted support this morning.",
    "Kindly forward this to {name}.", "Speak to {name} in {city}.", "Attn: {name}",
    "cc {name} on the reply", "Handover notes prepared by {name}.",
    "Our {title}, {name}, will join the call.", "Escalated to {name}.",
    "{name} and {name} both reported the issue.", "Ask {first} about it.",
    "Signed, {name}", "-- {name}, {title}", "Dr. {name} reviewed the file.",
    "The claim was filed by {name} on behalf of her son {first}.",
    # Field labels: the label itself becomes a candidate and must come out negative,
    # otherwise the model masks the word "Name" along with the name.
    "Name: {name}", "Full name: {name}", "Customer: {name}", "Patient: {name}",
    "Holder: {name}", "Applicant: {name}, {title}", "Beneficiary: {name}",
    "Name: {name} | Phone: {phone} | Email: {email}",
    "Customer: {name}\nOrder: {order}\nStatus: shipped",
]

CONTACT_PHRASINGS = [
    "Reach me on {phone}.", "My number is {phone}.", "Call {phone} after 6pm.",
    "You can text {phone} anytime.", "Mobile: {phone}", "Ph: {phone}",
    "Drop a mail to {email}.", "Email: {email}", "My email address is {email}.",
    "Write to {email} for the invoice.", "Contact: {email} / {phone}",
    "I live at {address}, {city} {pincode}.", "Deliver to {flat}, {city}.",
    "Billing address: {address}, {city}, {state} {pincode}.",
    "Shipping to {address} ({city}).", "Home address is {address}.",
    "Login as {username}.", "Username: {username}", "The user id is {username}.",
    "Account number {account} at the {city} branch.", "Credit the amount to account {account}.",
]

ID_PHRASINGS = [
    "My Aadhaar is {aadhaar}.", "Aadhaar number {aadhaar} was submitted.",
    "Aadhaar {aadhaar} is linked to mobile {phone}.", "PAN {pan} was rejected during KYC.",
    "My PAN card number is {pan}.", "Passport: {passport}", "Passport number {passport} expires soon.",
    "SSN {ssn} is on file.", "My Social Security Number is {ssn}.",
    "MRN {mrn} shows elevated glucose.", "Patient MRN: {mrn}",
    "Insurance member ID {member_id} covers the scan.", "Member ID: {member_id}",
    "Employee ID {emp_id} belongs to {name}.", "Emp id {emp_id} was deactivated.",
    "Policy {policy} renews next month.", "The policy number is {policy}.",
    "Account ID {account} was flagged.", "Vehicle registration {vehicle} is overdue.",
    "Driving record for {vehicle}.", "Tax ID {ssn} is registered to {name}.",
]

SECRET_PHRASINGS = [
    "Password: {password}", "The password is {password}, please change it.",
    "Temporary password {password} expires in 24h.", "API key {apikey} was leaked.",
    "The token is {apikey}.", "Your OTP is {otp}. Do not share it.",
    "OTP {otp} is valid for 5 minutes.", "CVV {cvv} was entered incorrectly.",
    "Card {card} declined; CVV {cvv}.", "UPI ID {upi} received the refund.",
    "Pay to {upi}.", "Login from {ip} using {username}.",
    "Device MAC {mac} joined the network.",
]

# HARD NEGATIVES: the value looks exactly like PII but refers to nothing personal.
HARD_NEGATIVES = [
    "Ticket {ticket} was raised for the {product} outage.",
    "Order {order} shipped from the {city} warehouse.",
    "{build} completed in {count} minutes.",
    "{room} is booked from {time}.",
    "Transaction ID {~account} completed successfully.",
    "Reference number {~account} was quoted on the invoice.",
    "Server IP {~ip} is behind the load balancer.",
    "The health check pings {~ip} every 30 seconds.",
    "Amount {amount} was credited on {month_date}.",
    "Version {version} shipped on {day}.",
    "Project code {~emp_id} tracks the migration budget.",
    "Cost centre {~emp_id} was closed last quarter.",
    "Test environment generated value {~phone} for the fixture.",
    "The sample dataset contains the dummy number {~phone}.",
    "Load test ran with user id {~account} for {count} iterations.",
    "Batch {~mrn} failed validation and was reprocessed.",
    "The HSN code is {hsn} and GST is {amount}.",
    "Support queue {ticket} has {count} open items.",
    "Commit {commit} fixed the null pointer in {product}.",
    "Listening on port {port}; see {ticket} for details.",
    "{sprint} closed with {count} story points.",
    "Invoice {order} total {amount} due {month_date}.",
    "The checksum {~mrn} did not match the manifest.",
    "Warehouse SKU {order} holds {count} units.",
    "Document {~pan} is the template identifier, not a tax number.",
    "Rate limit is {count} requests per minute from {~ip}.",
    "The {landmark} in {city} attracts {count} visitors a year.",
    "{company} and {company} opened offices in {city}.",
    "{product} was upgraded to {version} on {day}.",
    "The {city} team reported {amount} growth this quarter.",
    "Meeting notes: {sprint}, owner {title}, due {day}.",
    "Release {version} is tagged {commit} in the {product} repo.",
    "The report contains no personal information. Revenue was {amount}.",
    "Backup completed at {time} across {count} tables.",
    "Flight AI {count} departs {city} at {time}.",
    "{company} kept the repo rate at {amount}.",
]

# CONTRAST PAIRS: the same shape in both roles inside one text.
CONTRAST = [
    "Contact {name} at {phone}. The reference number {~phone} is not a contact.",
    "Call {phone} for support; ticket {ticket} tracks the issue.",
    "Employee {emp_id} is {name}; cost centre {~emp_id} is unrelated.",
    "The server at {~ip} was accessed by {name} from {ip}.",
    "Account {account} belongs to {name}. Order {order} is separate.",
    "Patient MRN {mrn} differs from batch id {~mrn}.",
    "Send the OTP {otp} to {phone}, not to queue {ticket}.",
    "{name} filed ticket {ticket} about invoice {order}.",
    "Transfer {amount} from account {account} to {name}.",
    "Room {~last4} is booked for {name}, whose card ends {last4}.",
]

EMAIL_BLOCKS = [
    "From: {name} <{email}>\nTo: support@{company}.example\nSubject: {ticket}\n\nHi team, I was charged {amount} twice. Regards, {first}",
    "Best regards,\n{name}\n{title}\n{email}\n{phone}",
    "Thanks,\n{first}\n{phone}",
    "Hi {first},\nPlease review {ticket} before {day}.\nThanks,\n{first}",
    "To: hr@{company}.example\nSubject: Leave\n\nI will be out from {month_date}. -- {name}",
]

CHAT = [
    "hey its {lower_first} here, my num is {phone}",
    "{lower_first} here! drop the parcel at {flat}, {city}",
    "yo its {lower_first}, text me at {phone} when ur free",
    "pls send the report to {email} asap thx",
    "call me on {phone}, im near {landmark}",
    "bro my aadhaar {aadhaar} got linked to wrong number lol",
    "hi this is {lower_first}, can u reschedule my appointment?",
    "ok so the meeting is at {time}, dont be late",
    "{lower_first} speaking, acct {account} pls check",
    "ugh {ticket} still open, any update?",
]

MULTILINGUAL = [
    "Mera naam {name} hai, mera phone {phone} hai aur main {city} mein rehti hoon.",
    "Mera naam {name} hai. Mera mobile {phone} hai aur mera PAN hai {pan}.",
    "Aapka OTP {otp} hai. Kisi ke saath share na karein.",
    "Bonjour, je m'appelle {name}. Mon email est {email}.",
    "Je m'appelle {name}, mon numéro est {phone}.",
    "Hola, me llamo {name} y mi correo es {email}.",
    "Mi nombre es {name} y vivo en {city}.",
    "Mein Name ist {name}, meine Telefonnummer ist {phone}.",
    "Namaste, main {name} bol raha hoon, mera number {phone} hai.",
    # Non-Latin script: the salutation and ordinary words must stay, the name and number must go.
    "प्रिय टीम, मेरा नाम {hi_name} है। मेरा फोन नंबर {phone} है।",
    "नमस्ते, मैं {hi_name} हूँ और मेरा ईमेल {email} है।",
    "मेरा नाम {hi_name} है और मेरा पता {address}, {city} है।",
    "प्रिय महोदय, कृपया {ticket} की स्थिति बताएं। धन्यवाद।",
    "यह रिपोर्ट {company} के लिए है। कोई व्यक्तिगत जानकारी नहीं है।",
    "என் பெயர் {ta_name}. என் தொலைபேசி எண் {phone}.",
    "வணக்கம், இந்த அறிக்கையில் தனிப்பட்ட தகவல் இல்லை.",
]

MEDICAL_HR_FINANCE = [
    "Patient: {name}, DOB {dob}, MRN {mrn}. Contact husband {first} at {phone}.",
    "Discharge summary for {name}: admitted {month_date}, MRN {mrn}.",
    "Dr. {name} prescribed medication for patient {name} on {day}.",
    "Emergency contact for {name} is {first} at {phone}.",
    "Offer letter for {name}, joining {month_date}, {title} at {company}.",
    "Background check for {name}: PAN {pan}, Aadhaar {aadhaar}.",
    "Resignation received from {name}, last working day {month_date}.",
    "Payroll for {emp_id} goes to account {account}.",
    "Card {card} expiring 11/27 was declined for {amount}.",
    "IFSC {ifsc}, account {account}, holder {name}.",
    "The loan application of {name} (PAN {pan}) was approved.",
    "KYC for {name}, DOB {dob}, PAN {pan} is pending.",
]

ALL = (NAME_PHRASINGS + CONTACT_PHRASINGS + ID_PHRASINGS + SECRET_PHRASINGS + HARD_NEGATIVES
       + CONTRAST + EMAIL_BLOCKS + CHAT + MULTILINGUAL + MEDICAL_HR_FINANCE)

# ------------------------------------------------------------------ surface noise

WRAPPERS = [
    "{t}", "{t}", "{t}", "{t}",                      # plain, most of the time
    "Note: {t}", "FYI - {t}", "> {t}", "({t})",
    "{t}\n\n--\nsent from my phone", "Update: {t}",
    "{t} Please confirm.", "Hi team,\n{t}\nThanks.",
]


def _typo(s):
    """One realistic slip: a doubled letter, a dropped space or a lowercased start."""
    if len(s) < 12:
        return s
    i = random.randrange(len(s) - 1)
    kind = random.random()
    if kind < 0.34 and s[i].isalpha():
        return s[:i] + s[i] * 2 + s[i:]
    if kind < 0.67 and " " in s[10:]:
        j = s.index(" ", 10)
        return s[:j] + s[j + 1:]
    return s[0].lower() + s[1:]


def _apply_noise(text, spans):
    """Wrap or scuff the text, shifting the labelled spans to match."""
    if random.random() < 0.30:
        w = random.choice(WRAPPERS)
        pre = w[:w.index("{t}")]
        text = w.replace("{t}", text)
        spans = [(a + len(pre), b + len(pre), t) for a, b, t in spans]
    if random.random() < 0.10:
        # Only scuff after the last labelled span, so offsets stay exact.
        cut = max([b for _, b, _ in spans], default=0)
        head, tail = text[:cut], text[cut:]
        text = head + _typo(tail)
    if random.random() < 0.08:
        text = text.rstrip(".") + random.choice(["", " ", "  ", "!!"])
    return text, spans


# Which PII category each slot belongs to. The generator knows this for free, so typed
# training labels cost nothing extra.
SLOT_TYPE = {
    **{s: "person" for s in ("name", "first", "lower_first", "lower_name", "upper_name",
                             "mixed_name", "hi_name", "ta_name")},
    "email": "email", "phone": "phone", "dob": "date",
    **{s: "address" for s in ("address", "flat", "pincode", "zip")},
    **{s: "id" for s in ("aadhaar", "pan", "ssn", "passport", "mrn", "member_id",
                         "emp_id", "policy", "vehicle")},
    **{s: "financial" for s in ("card", "last4", "account", "cvv", "upi")},
    **{s: "secret" for s in ("password", "otp", "apikey", "username")},
    **{s: "network" for s in ("ip", "mac")},
}


def generate(n: int, seed: int = 0) -> list[dict]:
    """n texts, each {"text", "pii_spans": [(start, end, type), ...]}.

    Only `{slot}` spans are labelled. `{~slot}` values are PII-shaped on purpose and are
    left unlabelled, which is what teaches context over format.
    """
    random.seed(seed)
    out = []
    for _ in range(n):
        tpl = random.choice(ALL)
        text, spans, cursor = "", [], 0
        for m in re.finditer(r"\{(~?)(\w+)\}", tpl):
            text += tpl[cursor:m.start()]
            decoy_shape, slot = bool(m.group(1)), m.group(2)
            value = (PII.get(slot) or DECOY[slot])()
            if slot in PII and not decoy_shape:
                spans.append((len(text), len(text) + len(value), SLOT_TYPE[slot]))
            text += value
            cursor = m.end()
        text += tpl[cursor:]
        text, spans = _apply_noise(text, spans)
        out.append({"text": text, "pii_spans": spans})
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ex = generate(n, seed=1)
    for e in ex:
        print(repr(e["text"]), "->", [(e["text"][a:b], t) for a, b, t in e["pii_spans"]])
    pos = sum(len(e["pii_spans"]) for e in ex)
    print(f"\n{len(ex)} texts, {pos} labelled PII spans, "
          f"{sum(1 for e in ex if not e['pii_spans'])} with none")
