SYSTEM_PROMPT = """You are an expert Computer Vision Annotator specialized in generating CLIP-compatible visual descriptions.

Your task is to produce TWO levels of representation:
(1) A rich internal understanding of visual structure (parts, materials, geometry).
(2) A FINAL COMPRESSED TEXT PROMPT that will be fed directly into a CLIP text encoder.

CRITICAL CONSTRAINT (MANDATORY):
- The final text prompt MUST be ≤ 77 CLIP tokens.
- Assume CLIP does NOT parse JSON, syntax, or logical relations.
- Only visual noun phrases and adjectives are useful.
- If the description exceeds the token budget, you MUST compress it by merging or dropping low-salience details.

CORE RULES:
1. USE ONLY VISIBLE TRAITS: color, shape, texture, material, and visually separable parts.
2. NO FULL SENTENCES in the final CLIP prompt.
3. NO VERBS, NO RELATIONAL CLAUSES (e.g., "connected to", "extending from").
4. ENCODE PART STRUCTURE IMPLICITLY using short noun phrases and adjectives.
5. PRIORITIZE DISCRIMINATIVE PARTS over fine-grained details.
6. ALWAYS RESERVE TOKEN SLOTS for:
   - Global shape
   - Dominant color/material
   - 3–6 key visual parts

Think of the output as a dense visual keyword embedding, not natural language.

"""

# Lưu ý: Các phần là JSON được dùng {{ }} để escape cho hàm .format()
# Chỉ {CLASS_NAME} dùng ngoặc đơn để điền giá trị.

USER_PROMPT_TEMPLATE = """
Describe the visual appearance of the class: "{CLASS_NAME}".

Use the optional variables if provided. Treat compound labels as a single visual concept.
Structure your response as a STRICTLY VALID JSON object following this schema:

{{
  "class_name": "string",      # e.g., "Mammal", "Vehicle", "Furniture", "Insect"
  "global_description": "string",    # Dense 40-60 word summary. MUST mention overall shape, dominant colors, and material/texture.
  "discriminative_attributes": [     # 5-7 short phrases distinguishing this class from similar ones.
    "string", ...
  ],
  "part_details": {{                 # DYNAMIC KEYS based on the object's domain.
    "part_name_1": "Detailed description of visual appearance (color, shape, texture)",
    "part_name_2": "..."
  }},
  "spatial_relations": [             # Geometric sentences for structural alignment.
    "Part A is positioned [relation] Part B",
    "Part C extends from Part D"
  ]
}}

### GUIDELINES FOR ROBUST GENERATION:
- **Part Selection**: Do not use fixed keys. If the class is "Laptop", use keys like "screen", "keyboard", "trackpad". If "Tiger", use "stripes", "head", "claws".
- **Visual Vocabulary**: Use words like "cylindrical", "matte", "glossy", "metallic", "furry", "rectangular", "spherical".
- **Spatial Relations**: Crucial for Graph/Gram Matrix matching. Be explicit about topology (e.g., "The monitor is hinged to the base").
Describe the visual appearance of the class: "{CLASS_NAME}".

You must output a SINGLE compressed CLIP-compatible text prompt.

OUTPUT FORMAT (STRICT):
- Plain text only
- Comma-separated visual phrases
- No JSON, no explanations

TOKEN BUDGET RULE:
- The output MUST fit within 77 CLIP tokens.
- If needed, compress by:
  (a) Merging related parts into one phrase
  (b) Removing redundant attributes
  (c) Replacing relations with adjectives (e.g., "roof-mounted", "upper", "elongated")

CONTENT PRIORITY ORDER:
1. Class name
2. Global shape and scale
3. Dominant colors and materials
4. Key visual parts (3–6 max)
5. Distinctive patterns or textures

FAILURE CONDITION:
If the output exceeds 77 tokens, the response is INVALID and must be regenerated shorter.

### EXEMPLAR 1 (Animal):
User Input: "class_name": "Least Auklet",
  "domain_category": "Bird",
  "global_description": "A small seabird with a compact, round body and short wings. Its plumage is a mix of dark brown and gray, with a pale underside. The head features a distinctive white patch behind the eye, and the beak is short and black. The legs are dark and webbed, adapted for swimming.",
  "discriminative_attributes": [
    "Small and round body",
    "White patch behind the eye",
    "Short wings for quick flight",
    "Short black beak",
    "Webbed dark legs"
  ],
  "part_details": {{
    "head": "Small with a distinct white patch behind the eye and dark feathers around the rest of the head.",
    "body": "Compact and round, covered with a mix of dark brown and gray feathers on the upper body, with a pale gray underside.",
    "beak": "Short, black, and slightly pointed, located at the front of the face.",
    "legs": "Short and webbed, dark in color, adapted for swimming and diving.",
    "wings": "Short, rounded, and dark-colored, suitable for quick, agile flight in the seabird environment."
  }},
  "spatial_relations": [
    "The white patch behind the eye is located just below the eye on the side of the head.",
    "The beak is positioned at the front of the head, short and curved downward.",
    "The wings are attached to the sides of the body, positioned slightly beneath the body for efficient flight.",
    "The legs are positioned under the body, extending slightly beyond the rear of the bird."
  ]
}}

### EXEMPLAR 2 (Animal):
User Input: CLASS_NAME="African Elephant"

JSON Output:
{{
  "class_name": "Parakeet Auklet",
  "domain_category": "Bird",
  "global_description": "A small seabird with a round, compact body and a distinctive, short, hook-shaped beak. Its plumage is predominantly dark with a slight sheen, and it features a bright, contrasting white patch behind the eye. The wings are short and rounded, and the legs are webbed for swimming.",
  "discriminative_attributes": [
    "Short, hook-shaped beak",
    "Distinctive white patch behind the eye",
    "Small, round body",
    "Dark plumage with a slight sheen",
    "Webbed dark legs"
  ],
  "part_details": {{
    "head": "Small and rounded, with a notable white patch behind the eye and dark feathers on the rest of the head.",
    "body": "Compact, round body with dark plumage, typically brown or gray, and a slightly glossy finish.",
    "beak": "Short, hooked, and dark-colored, positioned at the front of the face.",
    "legs": "Short and webbed, dark in color, designed for swimming and foraging.",
    "wings": "Short and rounded, dark in color with a slight sheen, suitable for quick bursts of flight."
  }},
  "spatial_relations": [
    "The white patch behind the eye is situated on the side of the head, just below the eye.",
    "The beak is positioned at the front of the head, slightly curved downwards.",
    "The wings extend from the sides of the body, relatively short and compact.",
    "The legs are positioned at the bottom of the body, slightly extending beyond the tail."
  ]
}}

### EXEMPLAR 3 (Animal):
User Input: CLASS_NAME="Sunflower"

JSON Output:
{{
  "class_name": "Rhinoceros Auklet",
  "domain_category": "Bird",
  "global_description": "A medium-sized seabird with a stocky, compact body and a distinctive horn-like protuberance on its beak. Its plumage is mostly dark, with a slightly lighter underside. The wings are short and rounded, and the legs are webbed and dark, adapted for swimming and diving.",
  "discriminative_attributes": [
    "Horn-like protuberance on the beak",
    "Dark plumage with a lighter underside",
    "Short, rounded wings",
    "Webbed dark legs",
    "Stocky, compact body shape"
  ],
  "part_details": {{
    "beak": "Short and stout, with a distinctive, horn-like protuberance at the top of the beak, usually white or pale in color.",
    "body": "Compact and stocky, with dark brown or black plumage on the upper body and a pale grayish underside.",
    "head": "Small with dark feathers, including a pronounced ridge on the beak.",
    "wings": "Short, rounded, and dark, providing agility for quick flights over short distances.",
    "legs": "Webbed, dark-colored legs located beneath the body, adapted for efficient swimming."
  }},
  "spatial_relations": [
    "The horn-like protuberance on the beak extends forward from the upper mandible.",
    "The wings are positioned at the sides of the body, compact and rounded for maneuverability.",
    "The legs are situated beneath the body, slightly extending beyond the rear.",
    "The beak is positioned at the front of the head, with the protuberance protruding forward."
  ]
}}

### EXEMPLAR 4:
User Input: CLASS_NAME="Brewer Blackbird"

JSON Output:
{{
  "class_name": "Brewer Blackbird",
  "domain_category": "Bird",
  "global_description": "A medium-sized songbird with glossy black plumage, exhibiting a slight iridescent sheen. The male has a bright yellow eye, while the female has a more muted appearance. Its body is slender, with a long, pointed tail, and the beak is sharp and conical.",
  "discriminative_attributes": [
    "Glossy black plumage with iridescence",
    "Bright yellow eye in males",
    "Long, pointed tail",
    "Sharp, conical beak",
    "Males are more colorful than females"
  ],
  "part_details": {{
    "head": "Small, with dark glossy feathers and a striking bright yellow eye in males. The female has a duller appearance.",
    "body": "Slender, with smooth black plumage that reflects light, creating an iridescent sheen, especially in males.",
    "beak": "Sharp, conical, and dark in color, well-adapted for picking seeds and small insects.",
    "tail": "Long and pointed, slightly angled at the tips, contributing to agile flight.",
    "legs": "Thin, dark, and sturdy, adapted for walking and perching."
  }},
  "spatial_relations": [
    "The wings are positioned along the sides of the body, helping with flight, and are folded close when at rest.",
    "The tail extends from the rear of the body, long and pointed for balance during flight.",
    "The beak is located at the front of the head, extending slightly downward.",
    "The yellow eye of the male is positioned on the side of the head, near the base of the beak."
  ]
}}
### EXEMPLAR 5 (Vehicle):
User Input: CLASS_NAME="ambulance"

JSON Output:
{{
  "class_name": "ambulance",
  "global_description": "Boxy vehicle with white body and red stripes, emergency lights on top, standard wheels, front windshield, and side doors for passengers.",
  "discriminative_attributes": [
    "White and red color",
    "Emergency lights on roof",
    "Boxy vehicle shape",
    "Side doors for passengers",
    "Standard wheels"
  ],
  "part_details": {{
    "lights": "Red and blue flashing emergency lights on roof",
    "body": "Boxy, mostly white with red stripes",
    "wheels": "Four standard wheels",
    "windows": "Front windshield and side windows",
    "doors": "Side doors for passengers"
  }},
  "spatial_relations": [
    "Lights on top of the vehicle",
    "Doors on sides of the body",
    "Wheels beneath body",
    "Windshield at front"
  ]
}}

### EXEMPLAR 6 (Amphibian):
User Input: CLASS_NAME="amphibian"

JSON Output:
{{
  "class_name": "amphibian",
  "global_description": "Small to medium-sized creature with smooth moist skin, usually green or brown, short limbs, webbed feet, bulging eyes, and a rounded body suitable for swimming and hopping.",
  "discriminative_attributes": [
    "Smooth, moist skin",
    "Green or brown color",
    "Short limbs",
    "Webbed feet",
    "Bulging eyes"
  ],
  "part_details": {{
    "head": "Rounded head with bulging eyes",
    "body": "Compact, rounded with smooth skin",
    "limbs": "Short legs, sometimes webbed",
    "feet": "Webbed for swimming",
    "skin": "Moist and smooth"
  }},
  "spatial_relations": [
    "Limbs attached to sides of body",
    "Feet at end of limbs",
    "Head on top of body"
  ]
}}

### EXEMPLAR 7 (Footwear):
User Input: CLASS_NAME="ankle_boot"

JSON Output:
{{
  "class_name": "ankle_boot",
  "global_description": "Short leather shoe covering the ankle, typically with laces or zippers, rounded toe, flat or small heel, sturdy sole, often in brown or black colors.",
  "discriminative_attributes": [
    "Covers ankle",
    "Leather material",
    "Rounded toe",
    "Flat or small heel",
    "Laces or zipper closure"
  ],
  "part_details": {{
    "upper": "Leather upper covering the foot and ankle",
    "sole": "Sturdy flat sole",
    "toe": "Rounded shape",
    "heel": "Small or flat",
    "closure": "Laces or zipper"
  }},
  "spatial_relations": [
    "Upper surrounds foot and ankle",
    "Sole beneath foot",
    "Heel at back",
    "Toe at front"
  ]
}}

### EXEMPLAR 8 (Insect):
User Input: CLASS_NAME="ant"

JSON Output:
{{
  "class_name": "ant",
  "global_description": "Small insect with segmented body, dark brown or black color, six legs, antennae, and a pair of mandibles on the head for carrying and biting.",
  "discriminative_attributes": [
    "Segmented body",
    "Dark brown or black",
    "Six legs",
    "Antennae",
    "Mandibles"
  ],
  "part_details": {{
    "head": "Small with antennae and mandibles",
    "thorax": "Middle segment connecting legs",
    "abdomen": "Rear segment, slightly larger",
    "legs": "Three pairs attached to thorax",
    "antennae": "Pair on head for sensing"
  }},
  "spatial_relations": [
    "Head at front, abdomen at rear",
    "Legs attached to thorax",
    "Antennae on head",
    "Mandibles at front of head"
  ]
}}

### EXEMPLAR 9 (Mammal):
User Input: CLASS_NAME="armadillo"

JSON Output:
{{
  "class_name": "armadillo",
  "global_description": "Small mammal with hard segmented armor, rounded body, short legs, pointed snout, tail extending behind, typically gray or brown, capable of curling into a ball.",
  "discriminative_attributes": [
    "Segmented armor",
    "Rounded body",
    "Short legs",
    "Pointed snout",
    "Tail extending behind"
  ],
  "part_details": {{
    "head": "Small with pointed snout",
    "body": "Rounded with hard segmented armor",
    "legs": "Short and sturdy",
    "tail": "Long, extending behind",
    "armor": "Hard plates covering body"
  }},
  "spatial_relations": [
    "Head at front, tail at rear",
    "Legs beneath body",
    "Armor covering body"
  ]
}}

"""
