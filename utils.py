
# STATE UTILS:
states = [
    ("Alabama", "AL"), ("Alaska", "AK"), ("Arizona", "AZ"), ("Arkansas", "AR"), ("California", "CA"), ("Colorado", "CO"),
    ("Connecticut", "CT"), ("Delaware", "DE"), ("District of Columbia", "DC"), ("Florida", "FL"), ("Georgia", "GA"),
    ("Hawaii", "HI"), ("Idaho", "ID"), ("Illinois", "IL"), ("Indiana", "IN"), ("Iowa", "IA"), ("Kansas", "KS"),
    ("Kentucky", "KY"), ("Louisiana", "LA"), ("Maine", "ME"), ("Maryland", "MD"), ("Massachusetts", "MA"),
    ("Michigan", "MI"), ("Minnesota", "MN"), ("Mississippi", "MS"), ("Missouri", "MO"), ("Montana", "MT"),
    ("Nebraska", "NE"), ("Nevada", "NV"), ("New Hampshire", "NH"), ("New Jersey", "NJ"), ("New Mexico", "NM"),
    ("New York", "NY"), ("North Carolina", "NC"), ("North Dakota", "ND"), ("Ohio", "OH"), ("Oklahoma", "OK"),
    ("Oregon", "OR"), ("Pennsylvania", "PA"), ("Rhode Island", "RI"), ("South Carolina", "SC"), ("South Dakota", "SD"),
    ("Tennessee", "TN"), ("Texas", "TX"), ("Utah", "UT"), ("Vermont", "VT"), ("Virginia", "VA"), ("Washington", "WA"),
    ("West Virginia", "WV"), ("Wisconsin", "WI"), ("Wyoming", "WY")
]

states.sort(key=lambda x: x[0])

# Indices to skip
skip_indices = {3, 7, 14, 43, 52}

# Create the dictionary
state_dict = {}
index = 1
for full_name, abbreviation in states:
    while index in skip_indices:
        index += 1
    state_dict[index] = abbreviation
    index += 1


# INDUSTRY UTILS:
industry_dict = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management and Remediation Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "99": "Industries not classified",
    "00": "Total for all sectors"
}

industry_dict_abbrev = {k.split('-')[0]:v for k,v in industry_dict.items()}

naics_codes = [code[0:2] for code in list(industry_dict.keys())[:-2]]

sic_to_naics_mapping = {
    '01': '11',  # Agriculture, Forestry, Fishing and Hunting
    '02': '11',  # Agriculture, Forestry, Fishing and Hunting
    '07': '11',  # Agriculture, Forestry, Fishing and Hunting
    '08': '11',  # Agriculture, Forestry, Fishing and Hunting
    '09': '11',  # Agriculture, Forestry, Fishing and Hunting
    '10': '21',  # Mining, Quarrying, and Oil and Gas Extraction
    '12': '21',  # Mining, Quarrying, and Oil and Gas Extraction
    '13': '21',  # Mining, Quarrying, and Oil and Gas Extraction
    '14': '21',  # Mining, Quarrying, and Oil and Gas Extraction
    '15': '23',  # Construction
    '16': '23',  # Construction
    '17': '23',  # Construction
    '20': '31-33',  # Manufacturing
    '21': '31-33',  # Manufacturing
    '22': '31-33',  # Manufacturing
    '23': '31-33',  # Manufacturing
    '24': '31-33',  # Manufacturing
    '25': '31-33',  # Manufacturing
    '26': '31-33',  # Manufacturing
    '27': '31-33',  # Manufacturing
    '28': '31-33',  # Manufacturing
    '29': '31-33',  # Manufacturing
    '30': '31-33',  # Manufacturing
    '31': '31-33',  # Manufacturing
    '32': '31-33',  # Manufacturing
    '33': '31-33',  # Manufacturing
    '34': '31-33',  # Manufacturing
    '35': '31-33',  # Manufacturing
    '36': '31-33',  # Manufacturing
    '37': '31-33',  # Manufacturing
    '38': '31-33',  # Manufacturing
    '39': '31-33',  # Manufacturing
    '40': '48-49',  # Transportation and Warehousing
    '41': '48-49',  # Transportation and Warehousing
    '42': '48-49',  # Transportation and Warehousing
    '43': '48-49',  # Transportation and Warehousing
    '44': '48-49',  # Transportation and Warehousing
    '45': '48-49',  # Transportation and Warehousing
    '46': '48-49',  # Transportation and Warehousing
    '47': '48-49',  # Transportation and Warehousing
    '48': '48-49',  # Transportation and Warehousing
    '49': '48-49',  # Transportation and Warehousing
    '50': '42',  # Wholesale Trade
    '51': '42',  # Wholesale Trade
    '52': '44-45',  # Retail Trade
    '53': '44-45',  # Retail Trade
    '54': '44-45',  # Retail Trade
    '55': '44-45',  # Retail Trade
    '56': '44-45',  # Retail Trade
    '57': '44-45',  # Retail Trade
    '58': '44-45',  # Retail Trade
    '59': '44-45',  # Retail Trade
    '60': '52',  # Finance and Insurance
    '61': '52',  # Finance and Insurance
    '62': '52',  # Finance and Insurance
    '63': '52',  # Finance and Insurance
    '64': '52',  # Finance and Insurance
    '65': '52',  # Finance and Insurance
    '67': '52',  # Finance and Insurance
    '70': '72',  # Accommodation and Food Services
    '72': '71',  # Arts, Entertainment, and Recreation
    '73': '54',  # Professional, Scientific, and Technical Services
    '75': '81',  # Other Services (except Public Administration)
    '76': '81',  # Other Services (except Public Administration)
    '78': '81',  # Other Services (except Public Administration)
    '79': '71',  # Arts, Entertainment, and Recreation
    '80': '62',  # Health Care and Social Assistance
    '81': '81',  # Other Services (except Public Administration)
    '82': '61',  # Educational Services
    '83': '62',  # Health Care and Social Assistance
    '84': '62',  # Health Care and Social Assistance
    '86': '81',  # Other Services (except Public Administration)
    '87': '54',  # Professional, Scientific, and Technical Services
    '88': '56',  # Administrative and Support and Waste Management and Remediation Services
    '89': '55',  # Management of Companies and Enterprises
    '91': '99',  # Public Administration
    '92': '99',  # Public Administration
    '93': '99',  # Public Administration
    '94': '99',  # Public Administration
    '95': '99',  # Public Administration
    '96': '99',  # Public Administration
    '97': '99',  # Public Administration
    '99': '99',  # Public Administration
}

def convert_sic_to_naics(sic_code):
    """
    Converts a SIC code to a NAICS code using the provided mapping.

    Args:
        sic_code (str): The SIC code to convert.

    Returns:
        str: The corresponding NAICS code, or None if no mapping is found.
    """
    return sic_to_naics_mapping.get(sic_code[:2], None)
