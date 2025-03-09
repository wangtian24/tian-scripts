SYNTHESIZER_FIRST_ASSISTANT_PROMPT = """Start chatting..."""

SYNTHESIZER_GENERATE_PERSONA_PROMPT = """Generate a JSON object representing persona, interests, and writing style. Some examples are below:

{{"persona": "layman", "interests": ["TMNT", "Japanese mythology", "romantic stories"], "style": "lowercase informal"}}
{{"persona": "programmer", "interests": ["lua", "batch scripting", "command line", "automation"], "style": "lowercase informal"}}
{{"persona": "sports fan", "interests": ["basketball", "Cincinnati", "Fab Five", "1991", "sports history"], "style": "casual"}}
{{"persona": "programmer", "interests": ["VBS scripting", "AI development", "self-replicating code"], "style": "informal"}}
{{"persona": "writer", "interests": ["fashion", "psychology", "narrative", "women's issues"], "style": "creative descriptive"}}
{{"persona": "adventurer", "interests": ["fantasy", "pets", "world-saving", "relationships"], "style": "casual storytelling"}}
{{"persona": "translator", "interests": ["japanese language", "english language", "linguistics"], "style": "formal"}}
{{"persona": "scientist", "interests": ["virology", "diagnostic techniques", "laboratory methods"], "style": "formal"}}
{{"persona": "artist", "interests": ["drawing", "fantasy", "club activities"], "style": "lowercase informal"}}
{{"persona": "programmer", "interests": ["C++", "video encoding", "data structures"], "style": "lowercase informal"}}
{{"persona": "young superhero", "interests": ["martial arts", "alien technology", "adventure", "friendship"], "style": "playful informal"}}
{{"persona": "sports fan", "interests": ["college football", "OU", "UT", "championship games"], "style": "casual"}}
{{"persona": "recent graduate", "interests": ["career development", "self-improvement", "networking"], "style": "formal"}}
{{"persona": "fan", "interests": ["cartoons", "anime", "character pairings"], "style": "casual"}}
{{"persona": "theologian", "interests": ["biblical studies", "philosophy", "linguistics"], "style": "formal"}}
{{"persona": "sports analyst", "interests": ["college football", "team statistics", "game strategy"], "style": "informal"}}
{{"persona": "sports journalist", "interests": ["sports media", "college athletics", "broadcasting", "drama in sports"], "style": "informal"}}
{{"persona": "layman", "interests": ["mystery", "animals"], "style": "informal"}}
{{"persona": "anime fan", "interests": ["dragon ball", "martial arts", "anime", "video games"], "style": "casual"}}
{{"persona": "programmer", "interests": ["web development", "encryption", "security"], "style": "formal"}}
{{"persona": "programmer", "interests": ["batch scripting", "obfuscation", "command line tools"], "style": "informal"}}
{{"persona": "scientist", "interests": ["oncology", "cell biology", "medical imaging"], "style": "formal"}}
{{"persona": "mother", "interests": ["parenting", "nether ecology", "warp trees"], "style": "casual"}}
{{"persona": "sports historian", "interests": ["Gulf Star Athletic Conference", "college athletics", "sports history"], "style": "informal"}}
{{"persona": "historian", "interests": ["antique collectibles", "cultural artifacts", "19th century history"], "style": "formal"}}
{{"persona": "sports analyst", "interests": ["NCAA basketball", "statistics", "team history"], "style": "formal"}}
{{"persona": "layman", "interests": ["jokes", "geography", "current events"], "style": "lowercase informal"}}
{{"persona": "layman", "interests": ["aquarium care", "pet fish", "home decoration"], "style": "lowercase informal"}}
{{"persona": "psychologist", "interests": ["mental health", "friendship", "emotional support"], "style": "conversational informal"}}
{{"persona": "young witch", "interests": ["potion making", "village protection", "swamp ecology"], "style": "casual and slightly frantic"}}
{{"persona": "music industry analyst", "interests": ["concert promotion", "ticket sales", "artist branding"], "style": "formal"}}
{{"persona": "scientist", "interests": ["algorithm theory", "computational complexity", "mathematics"], "style": "formal"}}
{{"persona": "political commentator", "interests": ["corporate politics", "social media", "cultural conflict"], "style": "formal"}}
{{"persona": "support technician", "interests": ["customer service", "technical troubleshooting", "software maintenance"], "style": "formal"}}
{{"persona": "sports journalist", "interests": ["football", "Trevor Lawrence", "college athletics"], "style": "informal"}}
{{"persona": "scientist", "interests": ["nuclear physics", "radioactive isotopes", "safety protocols"], "style": "formal"}}
{{"persona": "student", "interests": ["time management", "study techniques", "note-taking"], "style": "lowercase informal"}}
{{"persona": "scientist", "interests": ["MRI imaging", "oncology", "radiology", "rectal cancer", "biomarkers"], "style": "formal"}}
{{"persona": "writer", "interests": ["1960s culture", "gender identity", "theatrical performance"], "style": "narrative, descriptive"}}
{{"persona": "sports journalist", "interests": ["New Jersey Generals", "Mike Riley", "Canton, Ohio", "Tom Benson Hall of Fame Stadium"], "style": "informal"}}
{{"persona": "programmer", "interests": ["APIs", "OCR", "Python", "data privacy"], "style": "informal"}}
{{"persona": "game designer", "interests": ["storytelling", "game mechanics", "fantasy lore"], "style": "casual narrative"}}
{{"persona": "screenwriter", "interests": ["film", "storytelling", "character development"], "style": "conversational"}}
{{"persona": "layman", "interests": ["language learning", "Hebrew", "alphabets"], "style": "lowercase informal"}}
{{"persona": "historian", "interests": ["British monarchy", "paleontology", "religious studies", "comics"], "style": "formal"}}
{{"persona": "historian", "interests": ["ancient civilizations", "political systems", "cultural anthropology"], "style": "formal"}}
{{"persona": "layman", "interests": ["time travel", "philosophy", "current events"], "style": "casual"}}
{{"persona": "fantasy writer", "interests": ["world-building", "character development", "mythical creatures", "storytelling"], "style": "creative narrative"}}
{{"persona": "researcher", "interests": ["statistical analysis", "education", "questionnaire design"], "style": "formal"}}
{{"persona": "layman", "interests": ["wealth acquisition", "quick money schemes", "playful banter"], "style": "lowercase informal"}}
{{"persona": "scientist", "interests": ["chemistry", "oxidation states", "ionic compounds"], "style": "formal"}}
{{"persona": "writer", "interests": ["gender identity", "relationships", "escorts"], "style": "narrative, descriptive"}}
{{"persona": "gamer", "interests": ["Minecraft", "storytelling", "character development"], "style": "casual informal"}}
{{"persona": "writer", "interests": ["anime", "storytelling", "supernatural themes", "character development"], "style": "narrative, engaging, dramatic"}}
{{"persona": "e-commerce analyst", "interests": ["product forecasting", "inventory management", "supply chain optimization", "data analysis", "market trends"], "style": "formal"}}
{{"persona": "layman", "interests": ["statues", "mythology", "smartphones"], "style": "lowercase informal"}}
{{"persona": "game designer", "interests": ["game mechanics", "character development", "world-building"], "style": "casual narrative"}}
{{"persona": "journalist", "interests": ["college sports", "university policy", "student welfare"], "style": "formal"}}
{{"persona": "adventurer", "interests": ["exploration", "fantasy creatures", "storytelling"], "style": "casual informal"}}
{{"persona": "fanfiction writer", "interests": ["Super Robot Monkey Team Hyperforce Go!", "storytelling", "animation", "character development"], "style": "casual"}}
{{"persona": "creative writing expert", "interests": ["story structure", "character development", "narrative techniques"], "style": "formal"}}
{{"persona": "layman", "interests": ["law", "economics"], "style": "lowercase informal"}}
{{"persona": "scientist", "interests": ["seismology", "European Macroseismic Scale", "earthquake engineering"], "style": "formal"}}
{{"persona": "creative writer", "interests": ["urban planning", "sports", "fantasy", "sociology"], "style": "playful informal"}}
{{"persona": "warrior princess", "interests": ["justice", "diplomacy", "martial prowess"], "style": "dramatic and intense"}}
{{"persona": "game designer", "interests": ["storytelling", "character development", "game mechanics"], "style": "casual"}}
{{"persona": "customer support representative", "interests": ["customer satisfaction", "service improvement", "communication"], "style": "formal"}}
{{"persona": "gamer", "interests": ["monster hunting", "game lore", "character ages"], "style": "casual"}}
{{"persona": "layman", "interests": ["psychology", "music", "hip-hop"], "style": "lowercase informal"}}

Generate a single JSON object. Try to imitate the above examples but don't just repeat them; be slightly different. "persona" should be a single noun. Do not explain or add markup. Random seed: {seed}"""
