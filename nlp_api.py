import re
from typing import List, Dict, Any
import spacy
from rapidfuzz import fuzz, process
from word2number import w2n

nlp = spacy.load("en_core_web_sm")

# ---------------- Keyword Lists ----------------
EXPENSE_KEYWORDS = [
    "tapau", "makan", "minum", "belanja",

    # --- Core spending verbs ---
    "spend", "spent", "pay", "paid", "paying",
    "buy", "bought", "purchase", "purchased",
    "order", "ordered", "checkout",

    # --- Usage verbs ---
    "use", "used", "took", "take", "taken",

    # --- Fee-related ---
    "fee", "fees", "charges", "charge", "charged",
    "surcharge", "service fee", "processing fee",
    "admin fee", "atm fee", "maintenance fee",

    # --- Losing money ---
    "lose", "lost", "missing", "stolen",

    # --- Giving money away ---
    "give", "gave", "donate", "donated",
    "tip", "tipped", "charity", "sponsor", "sponsored",

    # --- Deduction-based ---
    "deduct", "deducted", "withdraw", "withdrawn", "cash out",

    # --- Transfers going OUT ---
    "transfer out", "transfer_out", "tf out",
    "banked out", "send money", "sent money",
    "sent rm", "paid rm", "settle debt", "settled",

    # --- Bills & utilities ---
    "pay bill", "paybills", "settle bill", "renew", "renewed",
    "subscription renewal", "auto debit", "autodebit",

    # --- Transport triggers ---
    "toll", "fare", "petrol", "fuel", "diesel",
    "parking", "grab", "uber", "ride", "bus fare",
    "train fare", "touch n go", "tng", "smarttag",

    # --- Delivery & platforms ---
    "grabfood", "foodpanda", "delivery fee",

    # --- Marketplace (expenses) ---
    "shopee", "lazada", "zalora", "shein",
    "temu", "aliexpress", "amazon",

    # --- Malaysian slang (expenses) ---
    "belanja", "tapau", "tapao", "tapaued",
    "beli", "bayar", "topap", "topap-ed",
    "makan", "minum", "bawak makan",

    # --- Outgoing refunds ---
    "refund_out", "return payment", "hutang bayar",
]

INCOME_KEYWORDS = [
    # --- Receiving verbs ---
    "receive", "received", "get", "got",
    "earn", "earned", "incoming", "credited",
    "credit", "credited rm",

    # --- Salary & wages ---
    "salary", "gaji", "wage", "payout", "payday",

    # --- Bonuses & allowances ---
    "bonus", "allowance", "incentive",
    "stipend", "commission", "overtime pay",
    "ot pay", "increment",

    # --- Gifts ---
    "gift", "given", "angpau", "angpao",
    "duit raya", "duit hadiah",

    # --- Business / side income ---
    "freelance", "side gig", "side job",
    "side income", "business income",
    "sales", "sell", "sold", "resell", "resold",
    "profit from sale",

    # --- Transfers IN ---
    "deposit", "banked in", "bank in",
    "transfer in", "transfer_in", "tf in",
    "duit masuk", "masuk duit",

    # --- Wallet top-ups (crediting user wallet) ---
    "topup bonus", "reload bonus",

    # --- Refunds & cashback ---
    "refund", "rebate", "cashback", "reward",
    "payback", "compensation", "reimbursement",

    # --- Investment income ---
    "dividend", "interest", "profit", "return",
    "roi", "capital gain",

    # --- Prize money ---
    "win", "won", "prize", "jackpot",
    "rewarded", "contest winning",

    # --- Scholarships ---
    "scholarship", "bursary", "grant",
]

# =========================
# ✅ PATTERN BOOSTERS (daily-use coverage)
# =========================

# Broad patterns that catch MANY daily variants without listing every word
EXPENSE_PATTERNS = [
    r"\b(bought|buy|purchase|paid|pay|spent|spend|order|ordered|checkout)\b",
    r"\b(bayar|beli|belanja|tapau|makan|minum|bungkus)\b",
    r"\b(topup|reload)\b.*\b(tng|touch ?n ?go|grab|shopee|steam|psn|apple|google)\b",
    r"\b(transfer|tf|send)\b.*\b(out|keluar)\b",
    r"\b(rent|sewa|utilities|bil|bill|internet|wifi|unifi|tnb|ptptn)\b",
    r"\b(toll|tol|parking|parkir|petrol|minyak|diesel|ron95|ron97|rfid|smarttag)\b",
]

INCOME_PATTERNS = [
    r"\b(receive|received|got|get|earn|earned|credited|credit)\b",
    r"\b(salary|gaji|allowance|bonus|commission|stipend)\b",
    r"\b(transfer|tf|bank)\b.*\b(in|masuk)\b",
    r"\b(refund|cashback|rebate|reimbursement)\b",
    r"\b(from)\b.*\b(parent|parents|dad|mum|mom|friend|boss|client)\b",
]

# Merchant/platform quick mapping
MERCHANT_MAP = {
    "food": [
        "mamak", "kopitiam", "warung", "gerai", "restoran", "restaurant", "cafe",
        "grabfood", "foodpanda", "tealive", "chatime", "starbucks",
        "mcd", "mcdonald", "kfc", "dominos", "pizza hut", "subway",
        "old town", "papparich", "secret recipe", "gigi coffee"
    ],
    "transport": [
        "tng", "touch n go", "touchngo", "rfid", "smarttag",
        "shell", "petronas", "petron", "bhp", "caltex",
        "rapidkl", "mrt", "lrt", "ktm", "komuter", "grab", "uber",
        "toll", "tol", "parking", "parkir", "minyak", "petrol", "ron95", "ron97"
    ],
    "shopping": ["shopee", "lazada", "zalora", "shein", "temu", "uniqlo", "h&m", "zara", "cotton on"],
    "bills": ["unifi", "tnb", "maxis", "digi", "celcom", "umobile", "netflix", "spotify", "tm", "broadband"],
}

CATEGORIES = {

    # =====================================================
    #                    FOOD & DRINK
    # =====================================================
    "food": [
        "tapau", "tapao", "tapaued",
        "makan", "minum", "air", "nasi", "lunch", "dinner", "breakfast", "water",

        # General
        "food", "meal", "eat", "snack", "snacks", "drinks",
        "drink", "beverage", "cafe", "restaurant",

        # Meals
        "breakfast", "lunch", "dinner", "brunch", "supper",

        # Malaysian foods
        "nasi lemak", "nasi goreng", "nasi ayam", "ayam penyet",
        "roti canai", "roti telur", "thosai", "chapati",
        "mee goreng", "mee kari", "laksa", "asam laksa",
        "char kuey teow", "char kway teow", "maggie goreng",
        "ayam goreng", "ikan goreng", "ramen", "udon", "sushi",
        "kopitiam", "kedai makan", "warung", "gerai", "mamak", "restoran",
        "nasi", "ayam", "ikan", "daging", "mee", "bihun", "kuey teow", "maggi",
        "soup", "sup", "goreng", "bakar", "kari", "sambal",
        "burger", "fries", "wrap", "nugget", "pizza", "pasta",
        "teh", "kopi", "ais", "panas", "sirap", "jus", "juice", "boba", "milk tea",
        "roti", "toast", "telur", "cheese", "curry puff",
        "kedai", "tapau", "bungkus",

        # Drinks
        "teh tarik", "teh ais", "kopi", "coffee", "milo",
        "bandung", "lime juice", "sirap ais",

        # Snacks & desserts
        "kuih", "ice cream", "dessert", "cake", "pastry",
        "donut", "bread", "bun", "croissant",

        # Chill/Popular chains
        "mcdonald", "mcd", "kfc", "burger king",
        "pizza hut", "dominos", "marrybrown",
        "subway", "starbucks", "dunkin",
        "secret recipe", "tealive", "chatime",
        "gigi coffee", "old town", "papparich",

        # Groceries
        "groceries", "grocery", "supermarket",
        "jaya grocer", "mydin", "aeon", "lotus",
        "giant", "tesco", "fresh market",

        # Delivery
        "grabfood", "foodpanda"
    ],

    # =====================================================
    #                      TRANSPORT
    # =====================================================
    "transport": [
        "transport", "grab", "uber", "e-hailing",
        "taxi", "bus", "train", "lrt", "mrt",
        "monorail", "erls", "rapidkl",

        "toll", "petrol", "fuel", "diesel",
        "parking", "fare", "ticket",

        "touch n go", "tng", "smarttag",
        "car", "motor", "bike", "motorcycle",
        "minyak", "isi minyak", "refuel", "pump", "pam",
        "ron95", "ron97", "diesel",
        "shell", "petronas", "petron", "bhp", "caltex",
        "parking", "park", "parkir", "kupon parking", "parking coupon",
        "toll", "tol", "duit tol", "smarttag", "rfid",
        "komuter", "ktm", "bus", "bas", "van", "carpool",
        "e-wallet fare", "ticket", "tiket", "rapid", "rapidkl"
    ],

    # =====================================================
    #                     SHOPPING / RETAIL
    # =====================================================
    "shopping": [
        "shopping", "mall", "retail",

        # Clothing
        "clothes", "clothing", "shoes", "fashion",
        "apparel", "jacket", "shirt", "pants",

        # Brands
        "uniqlo", "h&m", "zara", "cotton on",
        "puma", "nike", "adidas",

        # Online shopping
        "shopee", "lazada", "zalora", "shein",
        "temu", "aliexpress", "amazon",

        # Electronics
        "electronics", "gadget", "phone case",
        "earpods", "charger", "powerbank",
        "laptop", "keyboard", "mouse",
        "kedai", "store", "shop",
        "baju", "seluar", "tudung", "scarf", "stokin", "sock",
        "beg", "bag", "wallet", "dompet",
        "gift", "present",
        "accessory", "aksesori",
        "phone", "casing", "case", "screen protector", "tempered glass",
        "earphone", "earbuds", "headphone"
    ],

    # =====================================================
    #                        BILLS
    # =====================================================
    "bills": [
        "bill", "electric", "water", "wifi", "internet",
        "broadband", "tnb", "tm", "unifi",
        "celcom", "maxis", "digi", "umobile",

        "insurance", "insurance premium",
        "loan", "mortgage", "rent", "utilities",
        "ptptn", "tax", "quit rent",

        # Streaming
        "spotify", "netflix", "disney", "youtube premium",
        "apple music", "prime video", "bil", "bill", "bayar bil", "bayar bill",
        "topup", "reload", "prepaid", "postpaid",
        "data", "mobile data", "hotspot",
        "electric", "letrik", "air", "water",
        "sewa", "rent", "rental",
        "maintenance", "yuran", "fees", "subscription", "sub"
    ],

    # =====================================================
    #                 ENTERTAINMENT & LEISURE
    # =====================================================
    "entertainment": [
        "movie", "cinema", "concert", "festival", "music",
        "game", "gaming", "steam", "ps5", "playstation",
        "xbox", "nintendo", "mobile legends", "pubg",

        "bowling", "karaoke", "zoo", "museum",
        "theme park", "funfair", "escape room", "trampoline park",
        "arcade", "gym", "fitness", "yoga", "pilates", "sport", "olahraga",
        "futsal", "badminton", "tennis", "golf", "swimming", "renang", "basketball", "sepak bola", "football",
        "soccer", "volleyball", "hiking", "trekking", "outdoors", "camping", "fishing", "memancing", "gymnasium",
    ],

    # =====================================================
    #                      HEALTHCARE
    # =====================================================
    "healthcare": [
        "clinic", "hospital", "doctor", "pharmacy",
        "medicine", "prescription", "vitamin",
        "watsons", "guardian", "first aid",
        "covid test", "medical checkup", "ubat", "medicine", "med", "panadol", "paracetamol",
        "mc", "clinic fee", "checkup", "dental", "dentist", "optical", "eye care", "glasses", "contact lens"
    ],

    # =====================================================
    #                      EDUCATION
    # =====================================================
    "education": [
        "school", "tuition", "university",
        "course", "fees", "exam fees",
        "stationery", "book", "notebook",
        "textbook", "pen", "pencil", "assignment", "kelas", "class", "lecture", "lab",
        "notes", "nota", "printing", "print", "photostat", "xerox",
        "college", "mmu", "campus", "library",
        "file", "folder", "binder", "stapler", "scissors"
    ],

    # =====================================================
    #                       BANKING
    # =====================================================
    "banking": [
        "bank fee", "service charge", "processing fee",
        "atm fee", "withdrawal fee", "transfer fee",
        "exchange fee", "foreign exchange fee",
    ],

    # =====================================================
    #                    PERSONAL CARE
    # =====================================================
    "personal_care": [
        "salon", "haircut", "barber", "spa",
        "massage", "skincare", "soap",
        "shampoo", "perfume", "makeup", "cosmetics"
    ],

    # =====================================================
    #                         PETS
    # =====================================================
    "pets": [
        "pet food", "cat food", "dog food",
        "vet", "pet shop", "grooming",
        "cat litter", "pet accessories", "pet supplies", "pet care", "pet clinic",
        "pet hospital", "veterinary", "veterinarian", "pet medication", "pet vaccine"
    ],

    # =====================================================
    #                      HOME SUPPLIES
    # =====================================================
    "home": [
        "ikea", "furniture", "home decor",
        "cleaning supplies", "detergent",
        "dish soap", "broom", "mop",
    ],

    "income": [
        "salary", "gaji", "gift", "bonus", "allowance", "commission",
        "rebate", "refund", "cashback", "bank in", "duit masuk",
        "deposit", "earned", "interest", "dividend", "profit",
        "sell", "sold", "payment received",
        "parent", "parents"
    ],

    # =====================================================
    #                     OTHER / UNKNOWN
    # =====================================================
    "other": [
        "other", "misc", "miscellaneous",
        "uncategorized"
    ]
}

CATEGORY_KEYWORDS_FLAT = {cat: kws for cat, kws in CATEGORIES.items()}


def lemmatize(text: str) -> List[str]:
    """Return lemmas for tokens in the text (lowercased)."""
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]


def _normalize_thousands_commas(text: str) -> str:
    return re.sub(r'(?<=\d),(?=\d)', '', text)

NUM_WORDS = set("""
zero one two three four five six seven eight nine ten
eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen
twenty thirty forty fifty sixty seventy eighty ninety
hundred thousand million billion
and
""".split())

def _is_num_word(tok: str) -> bool:
    tok = re.sub(r"[^a-z]", "", tok.lower())
    return tok in NUM_WORDS

def _words_to_numbers(text: str) -> str:
    """
    Converts any contiguous number-word phrase anywhere in the sentence into digits.
    Examples:
      "spent five on food" -> "spent 5 on food"
      "grab two hundred and ten" -> "grab 210"
      "i got three thousand five hundred" -> "i got 3500"
    """
    if not text:
        return text

    # normalize currency words (optional)
    t = re.sub(r"\b(ringgit|myr)\b", "rm", text.lower())

    tokens = t.split()
    out = []
    i = 0

    while i < len(tokens):
        if not _is_num_word(tokens[i]):
            out.append(tokens[i])
            i += 1
            continue

        # build the longest possible number phrase from here
        j = i
        phrase_tokens = []
        while j < len(tokens) and _is_num_word(tokens[j]):
            phrase_tokens.append(tokens[j])
            j += 1

        phrase = " ".join(phrase_tokens)
        phrase_clean = re.sub(r"[^a-z\s-]", " ", phrase)
        phrase_clean = re.sub(r"\s+", " ", phrase_clean).strip()

        try:
            val = w2n.word_to_num(phrase_clean)
            out.append(str(val))
        except:
            # if conversion fails, keep original tokens
            out.extend(phrase_tokens)

        i = j

    return " ".join(out)

# =========================
# ✅ Improved amount extractor (keeps your logic + adds rm suffix, ringgit, k)
# =========================
def extract_amount(text: str) -> List[float]:
    if not text or not text.strip():
        return []

    t = text.lower()
    t = t.replace("₨", " rm ").replace("myr", " rm ").replace("rm", " rm ").replace("ringgit", " rm ")
    t = t.replace("$", " rm ")
    t = _normalize_thousands_commas(t)
    t = _words_to_numbers(t)

    matches = []

    # rm 12.50 / rm12.50
    for m in re.finditer(r'\brm\s*([+-]?\d+(?:\.\d{1,2})?)\b', t, flags=re.IGNORECASE):
        try:
            matches.append((m.start(), float(m.group(1))))
        except ValueError:
            pass

    # 12.50 rm / 12rm
    for m in re.finditer(r'\b([+-]?\d+(?:\.\d{1,2})?)\s*rm\b', t, flags=re.IGNORECASE):
        try:
            matches.append((m.start(), float(m.group(1))))
        except ValueError:
            pass

    # 10k / 2.5k
    for m in re.finditer(r'\b([+-]?\d+(?:\.\d+)?)\s*k\b', t, flags=re.IGNORECASE):
        try:
            matches.append((m.start(), float(m.group(1)) * 1000))
        except ValueError:
            pass

    # fallback numbers (avoid duplicates near rm matches)
    for m in re.finditer(r'(?<![\w.])([+-]?\d+(?:\.\d{1,2})?)', t):
        start = m.start()
        if any(abs(start - p) < 6 for p, _ in matches):
            continue
        try:
            matches.append((start, float(m.group(1))))
        except ValueError:
            pass

    matches.sort(key=lambda x: x[0])

    seen = set()
    result = []
    for _, val in matches:
        if val not in seen:
            result.append(val)
            seen.add(val)

    return result


def detect_transaction_type(text: str) -> str:
    """Detect whether the text indicates 'income', 'expense', or 'unknown'."""
    lemmas = lemmatize(text)
    txt = text.lower()

    # ✅ Pattern-first boost (daily phrasing)
    if any(re.search(p, txt) for p in INCOME_PATTERNS):
        return "income"
    if any(re.search(p, txt) for p in EXPENSE_PATTERNS):
        return "expense"

    # FIX: If sentence says "give/gave me" → INCOME
    if re.search(r"\b(giv(?:e|en|ing)|gave)\b.*\bme\b", txt):
        return "income"

    # FIX: "from friend/parents" → income
    if "from" in txt and any(word in txt for word in ["friend", "parents", "parent", "dad", "mum", "mom", "bro", "sis"]):
        return "income"

    # If I am the giver → EXPENSE
    if re.search(r"\bi\b.*\b(giv(?:e|en|ing)|gave)\b", txt):
        return "expense"

    # If "give/gave" + target person → EXPENSE
    if re.search(r"\b(giv(?:e|en|ing)|gave)\b\s+(my|parent|parents|mum|mom|dad|friend|bro|sis|brother|sister)\b", txt):
        return "expense"

    if any(word in txt for word in ["spent", "spend"]):
        return "expense"

    for tok in re.findall(r"[A-Za-z]+", txt):
        fuzzy_give = process.extractOne(tok, ["give", "gave", "given", "giving"], scorer=fuzz.ratio)
        if fuzzy_give and fuzzy_give[1] >= 80:
            # Fuzzy "give me" → INCOME
            if "me" in txt:
                return "income"

            # Fuzzy "I give" → EXPENSE
            if re.search(r"\bi\b", txt):
                return "expense"

            # Fuzzy "give" + target → EXPENSE
            if any(person in txt for person in ["parent", "parents", "mum", "mom", "dad", "friend", "bro", "sis", "brother", "sister"]):
                return "expense"

    verb_tokens = re.findall(r"[A-Za-z]+", txt)

    # Fuzzy income verbs
    for tok in verb_tokens:
        match = process.extractOne(tok, INCOME_KEYWORDS, scorer=fuzz.ratio)
        if match and match[1] >= 80:
            return "income"

    # Fuzzy expense verbs
    for tok in verb_tokens:
        match = process.extractOne(tok, EXPENSE_KEYWORDS, scorer=fuzz.ratio)
        if match and match[1] >= 80:
            return "expense"

    has_expense = any(w in lemmas or re.search(rf"\b{re.escape(w)}\b", txt) for w in EXPENSE_KEYWORDS)
    has_income = any(w in lemmas or re.search(rf"\b{re.escape(w)}\b", txt) for w in INCOME_KEYWORDS)

    if has_expense and has_income:
        # Prefer expense when explicit expense verbs present
        if any(re.search(rf"\b{re.escape(w)}\b", txt) for w in ["pay", "paid", "spend", "spent", "buy", "bought", "gave"]):
            return "expense"
        return "income"

    if has_expense:
        return "expense"
    if has_income:
        return "income"

    return "unknown"


def match_category(token: str) -> str:
    """
    Fuzzy-match a token against known category keywords.
    Returns category name if confident, else 'other'.
    """
    token = token.lower()
    best_cat = "other"
    best_score = 0
    for cat, keywords in CATEGORY_KEYWORDS_FLAT.items():
        match = process.extractOne(token, keywords, scorer=fuzz.ratio)
        if match:
            _, score, _ = match
            if score > best_score and score >= 78:
                best_score = score
                best_cat = cat
    return best_cat


def detect_category(text: str, transaction_type: str) -> str:
    """
    Determine a category by trying:
    - merchant/brand quick match (daily use)
    - phrase overrides (kedai makan, isi minyak)
    - exact lemma/key matching
    - phrase (ngram) fuzzy match
    - fuzzy token matching fallback
    """
    text_lemmas = lemmatize(text)
    txt = text.lower()

    # Prefer income categories for income messages
    if transaction_type == "income":
        possible = {"income": CATEGORIES["income"]}
    else:
        possible = {k: v for k, v in CATEGORIES.items() if k != "income"}

    # ✅ Phrase-first overrides (reduces "other" and wrong category)
    if transaction_type != "income":
        if any(p in txt for p in ["kedai makan", "mamak", "warung", "kopitiam", "gerai", "restoran"]):
            return "food"
        if any(p in txt for p in ["isi minyak", "ron95", "ron97", "petrol", "minyak", "diesel", "rfid", "smarttag"]):
            return "transport"

    # ✅ Merchant mapping (fast and strong)
    for cat, words in MERCHANT_MAP.items():
        if cat in possible and any(w in txt for w in words):
            return cat

    # Exact or lemma-based match
    for cat, keys in possible.items():
        for k in keys:
            if k in text_lemmas or re.search(rf"\b{re.escape(k)}\b", txt):
                return cat

    # ✅ Phrase scan (2-4 word ngrams) for "touch n go", "nasi lemak", etc.
    words = re.findall(r"[a-z0-9]+", txt)
    ngrams = []
    for n in (2, 3, 4):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i + n]))

    for cat, keys in possible.items():
        for k in keys:
            if " " in k:
                if k in txt:
                    return cat
                if ngrams:
                    best = process.extractOne(k, ngrams, scorer=fuzz.ratio)
                    if best and best[1] >= 85:
                        return cat

    # Fuzzy fallback: check each token
    tokens = re.findall(r"[A-Za-z0-9]+", txt)
    for tok in tokens:
        cat = match_category(tok)
        if cat != "other":
            return cat

    # Default
    return "income" if transaction_type == "income" else "other"


# ---------------- Public API ----------------

def analyze_single_message(text: str) -> List[Dict[str, Any]]:
    """
    Analyze a single short message and return a list of one-or-more
    transaction dicts: {type, amount, category}
    """
    ttype = detect_transaction_type(text)
    amounts = extract_amount(text)
    category = detect_category(text, ttype)

    if ttype == "unknown" and amounts:
        if category == "income":
            ttype = "income"
        else:
            ttype = "expense"

    if not amounts:
        return [{"type": ttype, "amount": None, "category": category}]

    return [{"type": ttype, "amount": float(a), "category": category} for a in amounts]


def analyze_message(text: str) -> List[Dict[str, Any]]:
    """
    Split possibly multi-transaction text into parts, analyze each, and
    return a concatenated list of transaction dicts.
    Splitting logic tries to split on ';', ' and ', ' & ', ' also ', newlines and commas after numbers.
    """
    if not text or not text.strip():
        return []

    separators = r'(?:\s*;\s*|\s+\band\b\s+|\s+\&\s+|\s+\balso\b\s+|\n|(?<=\d),\s+| \/\s+)'
    parts = [p.strip() for p in re.split(separators, text, flags=re.IGNORECASE) if p.strip()]

    results = []
    for p in parts:
        results.extend(analyze_single_message(p))
    return results


# ===========================
# ✅ ADD THIS AT THE VERY END
# ===========================

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS: allow browser (Flutter Web) to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:51877",
        "http://127.0.0.1:51877",
        "http://localhost:*",
        "http://127.0.0.1:*",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return {"results": analyze_message(req.text)}