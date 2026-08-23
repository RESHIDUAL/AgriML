"""
disease_info.py - Agricultural Knowledge Base for PlantDoc Target Classes (Strategy A)

Contains detailed disease information for all 27 unified class names used in the
AgriML plant disease classification system. Each entry includes symptoms, causes,
prevention tips, chemical and organic treatment recommendations, and severity ratings.

All pathogen names and treatment recommendations are based on established
agricultural science references.
"""

DISEASE_DB: dict[str, dict] = {
    # -------------------------------------------------------------------------
    # APPLE
    # -------------------------------------------------------------------------
    "Apple Scab": {
        "crop": "Apple",
        "disease": "Scab",
        "is_healthy": False,
        "symptoms": (
            "Olive-green to dark brown velvety lesions appear on leaves, often along "
            "veins. Infected fruit develop raised, corky, dark scab-like spots that may "
            "crack. Severe infections cause premature leaf drop and deformed fruit."
        ),
        "causes": (
            "Caused by the fungus Venturia inaequalis. The pathogen overwinters in "
            "fallen infected leaves and releases ascospores during wet spring weather. "
            "Prolonged leaf wetness and cool temperatures (15-20 °C) favour infection."
        ),
        "prevention": [
            "Rake and destroy fallen leaves in autumn to reduce overwintering inoculum.",
            "Plant scab-resistant apple cultivars (e.g., Liberty, Enterprise, Prima).",
            "Ensure adequate spacing and pruning for good air circulation.",
            "Apply protective fungicide sprays from green-tip through petal fall.",
        ],
        "chemical_treatment": [
            "Captan 50 WP applied at 2.5-3.0 g/L during primary scab season.",
            "Myclobutanil (Rally) at labelled rates as a systemic curative option.",
            "Mancozeb 75 WP at 2.5 g/L as a protectant spray in tank-mix programs.",
        ],
        "organic_treatment": [
            "Sulfur-based fungicides (e.g., wettable sulfur at 5-8 g/L) applied preventively.",
            "Neem oil (0.5-1 %) sprayed at 7-10 day intervals during wet periods.",
            "Copper hydroxide applied at green-tip before bloom to reduce early inoculum.",
        ],
        "severity": "Medium",
    },
    "Apple Cedar Rust": {
        "crop": "Apple",
        "disease": "Cedar Rust",
        "is_healthy": False,
        "symptoms": (
            "Bright yellow-orange spots appear on upper leaf surfaces, enlarging into "
            "lesions with a red border. The undersides of leaves develop tube-like "
            "structures (aecia) that release spores. Fruit may also show raised, "
            "orange-brown lesions."
        ),
        "causes": (
            "Caused by the fungus Gymnosporangium juniperi-virginianae. The pathogen "
            "requires an alternate host (Eastern red cedar or juniper) to complete its "
            "life cycle. Warm, wet spring weather triggers spore release from cedar galls."
        ),
        "prevention": [
            "Remove nearby cedar and juniper trees within a 2-mile radius where feasible.",
            "Plant cedar-rust-resistant apple cultivars (e.g., Redfree, Liberty).",
            "Apply fungicides from pink bud through 2-3 weeks after petal fall.",
            "Monitor local cedar trees for galls and remove them before spring.",
        ],
        "chemical_treatment": [
            "Myclobutanil (Rally) applied at pink, bloom, and petal fall stages.",
            "Propiconazole (Tilt) at labelled rates during the infection period.",
            "Mancozeb 75 WP at 2.5 g/L as a protectant applied before rain events.",
        ],
        "organic_treatment": [
            "Sulfur-based fungicides applied preventively at 7-10 day intervals.",
            "Neem oil sprays (0.5-1 %) to reduce spore germination on leaf surfaces.",
            "Bordeaux mixture (copper sulfate + lime) applied at green-tip stage.",
        ],
        "severity": "Medium",
    },
    "Apple Healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Maintain balanced fertilisation with emphasis on potassium and calcium.",
            "Ensure regular pruning to maintain open canopy and air circulation.",
            "Water at the base of the tree to keep foliage dry.",
            "Monitor trees weekly for early signs of pest or disease pressure.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # BELL PEPPER
    # -------------------------------------------------------------------------
    "Bell Pepper Bacterial Spot": {
        "crop": "Bell Pepper",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "symptoms": (
            "Small, dark, water-soaked lesions appear on leaves that become raised and "
            "scab-like. Spots may coalesce, causing leaves to yellow and drop. Fruit "
            "develop raised, rough, brown spots that reduce marketability."
        ),
        "causes": (
            "Caused by the bacterium Xanthomonas campestris pv. vesicatoria (also "
            "classified as X. euvesicatoria). The pathogen spreads via contaminated "
            "seed, infected transplants, splashing rain, and overhead irrigation. Warm, "
            "humid conditions (24-30 °C) favour disease development."
        ),
        "prevention": [
            "Use certified disease-free seed and transplants.",
            "Avoid overhead irrigation; use drip irrigation instead.",
            "Rotate crops with non-solanaceous plants for at least 2-3 years.",
            "Sanitise tools and equipment between fields.",
            "Remove and destroy infected plant debris after harvest.",
        ],
        "chemical_treatment": [
            "Copper hydroxide (Kocide 3000) at 1.0-1.75 g/L as a protectant.",
            "Copper hydroxide + Mancozeb tank-mix for improved efficacy.",
            "Acibenzolar-S-methyl (Actigard) as a systemic acquired resistance inducer.",
        ],
        "organic_treatment": [
            "Fixed copper sprays (copper octanoate) applied at 7-day intervals.",
            "Bacillus subtilis-based biofungicide (Serenade) for bacterial suppression.",
            "Compost tea foliar sprays to boost beneficial microbial populations.",
        ],
        "severity": "High",
    },
    "Bell Pepper Healthy": {
        "crop": "Bell Pepper",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Maintain consistent watering with drip irrigation.",
            "Provide balanced N-P-K fertilisation throughout the growing season.",
            "Mulch around plants to conserve moisture and suppress weeds.",
            "Scout plants regularly for early pest and disease detection.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # BLUEBERRY
    # -------------------------------------------------------------------------
    "Blueberry Healthy": {
        "crop": "Blueberry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Maintain soil pH between 4.5 and 5.5 with sulfur amendments as needed.",
            "Apply pine bark or sawdust mulch to conserve moisture and maintain acidity.",
            "Prune older canes annually to encourage vigorous new growth.",
            "Provide netting during fruiting to protect from bird damage.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # CHERRY
    # -------------------------------------------------------------------------
    "Cherry Healthy": {
        "crop": "Cherry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Prune trees to maintain an open vase shape for airflow and light penetration.",
            "Apply dormant-season copper spray to reduce overwintering pathogen load.",
            "Avoid mechanical damage to bark which invites canker pathogens.",
            "Ensure well-drained soil to prevent root rot.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # CORN (MAIZE)
    # -------------------------------------------------------------------------
    "Corn Gray Leaf Spot": {
        "crop": "Corn",
        "disease": "Gray Leaf Spot",
        "is_healthy": False,
        "symptoms": (
            "Rectangular, gray to tan lesions develop on lower leaves, expanding parallel "
            "to leaf veins. Lesions may coalesce, killing large areas of leaf tissue. "
            "Severe infections reduce photosynthetic capacity and grain fill."
        ),
        "causes": (
            "Caused by the fungus Cercospora zeae-maydis. The pathogen overwinters in "
            "infected corn debris on the soil surface. Prolonged humidity (>95 %), warm "
            "temperatures (22-30 °C), and reduced tillage favour disease development."
        ),
        "prevention": [
            "Plant resistant or tolerant corn hybrids.",
            "Rotate with non-host crops such as soybeans for at least one year.",
            "Incorporate or bury crop residue through tillage to reduce inoculum.",
            "Avoid planting corn-on-corn in fields with a history of the disease.",
        ],
        "chemical_treatment": [
            "Azoxystrobin (Quadris) applied at VT/R1 stage for protective activity.",
            "Pyraclostrobin + Metconazole (Headline AMP) at tasseling.",
            "Propiconazole (Tilt) at 1.0 mL/L as an early curative option.",
        ],
        "organic_treatment": [
            "Crop rotation and residue management are the primary organic strategies.",
            "Trichoderma-based biofungicides applied to soil to accelerate residue decomposition.",
            "Ensure wide row spacing and optimal plant population for air movement.",
        ],
        "severity": "High",
    },
    "Corn Common Rust": {
        "crop": "Corn",
        "disease": "Common Rust",
        "is_healthy": False,
        "symptoms": (
            "Small, circular to elongate, cinnamon-brown pustules (uredinia) form on "
            "both leaf surfaces. Pustules rupture to release powdery, reddish-brown "
            "spores. Heavy infections cause chlorosis and premature leaf senescence."
        ),
        "causes": (
            "Caused by the fungus Puccinia sorghi. Urediniospores are wind-dispersed "
            "from southern regions or tropical areas each season. Cool to moderate "
            "temperatures (16-23 °C) and high humidity promote rapid disease development."
        ),
        "prevention": [
            "Plant rust-resistant corn hybrids with Rp genes.",
            "Early planting to avoid peak spore arrival periods.",
            "Monitor fields regularly during the mid-vegetative growth stages.",
            "Scout for early pustule formation and apply fungicides if warranted.",
        ],
        "chemical_treatment": [
            "Azoxystrobin (Quadris) at 0.6-0.8 mL/L applied at first sign of pustules.",
            "Propiconazole (Tilt) at 1.0 mL/L for curative and protectant activity.",
            "Mancozeb 75 WP at 2.5 g/L as a protectant in high-risk areas.",
        ],
        "organic_treatment": [
            "Plant resistant hybrids - the most effective organic strategy.",
            "Sulfur-based fungicide sprays at early infection stages.",
            "Maintain crop vigour through balanced organic fertilisation.",
        ],
        "severity": "Medium",
    },
    "Corn Northern Leaf Blight": {
        "crop": "Corn",
        "disease": "Northern Leaf Blight",
        "is_healthy": False,
        "symptoms": (
            "Long, elliptical, gray-green to tan cigar-shaped lesions (2.5-15 cm) "
            "develop on lower leaves first and progress upward. Lesions may coalesce "
            "to blight entire leaves. Under humid conditions, dark spore production "
            "gives lesions a dusty appearance."
        ),
        "causes": (
            "Caused by the fungus Exserohilum turcicum (syn. Setosphaeria turcica). "
            "The pathogen overwinters in infected corn residue. Moderate temperatures "
            "(18-27 °C), heavy dew, and frequent rainfall promote infection cycles."
        ),
        "prevention": [
            "Plant hybrids with Ht resistance genes (Ht1, Ht2, Ht3, HtN).",
            "Rotate crops and manage residue to reduce inoculum carry-over.",
            "Avoid late planting which exposes crops during peak spore production.",
            "Maintain balanced fertility to support plant vigour and resistance.",
        ],
        "chemical_treatment": [
            "Pyraclostrobin (Headline) applied at or just before tasseling (VT).",
            "Azoxystrobin + Propiconazole (Quilt Xcel) at VT/R1 stage.",
            "Mancozeb 75 WP at 2.5 g/L as an early-season protectant.",
        ],
        "organic_treatment": [
            "Crop rotation with soybeans or small grains for at least one season.",
            "Trichoderma harzianum-based bio-agents applied to soil and residues.",
            "Maintain adequate plant spacing for improved canopy ventilation.",
        ],
        "severity": "High",
    },
    # -------------------------------------------------------------------------
    # GRAPE
    # -------------------------------------------------------------------------
    "Grape Black Rot": {
        "crop": "Grape",
        "disease": "Black Rot",
        "is_healthy": False,
        "symptoms": (
            "Circular, tan leaf spots with dark borders appear, often with tiny black "
            "pycnidia in the centre. Infected berries turn brown, shrivel into hard, "
            "black mummified fruit. Tendrils and shoots may also show dark, elongated "
            "lesions."
        ),
        "causes": (
            "Caused by the fungus Guignardia bidwellii (anamorph: Phyllosticta "
            "ampelicida). The pathogen overwinters in mummified berries and infected "
            "cane tissue. Warm, wet weather (20-27 °C) during the growing season "
            "promotes spore release and infection."
        ),
        "prevention": [
            "Remove and destroy mummified berries and infected cane prunings.",
            "Train vines on trellis systems to improve air circulation and sun exposure.",
            "Apply fungicides from bud break through 4-5 weeks after bloom.",
            "Avoid overhead irrigation; keep fruit zone dry.",
            "Maintain good canopy management with timely shoot positioning and hedging.",
        ],
        "chemical_treatment": [
            "Myclobutanil (Rally) applied from immediate pre-bloom through 4 weeks post-bloom.",
            "Mancozeb 75 WP at 2.0-2.5 g/L as a protectant during early season.",
            "Captan 50 WP at 2.5 g/L tank-mixed with a sterol inhibitor for broad-spectrum control.",
        ],
        "organic_treatment": [
            "Sulfur sprays applied at 7-10 day intervals from bud break to veraison.",
            "Copper-based fungicides (Bordeaux mixture) at early season.",
            "Thorough sanitation: remove all mummified fruit and prune out infected wood.",
        ],
        "severity": "High",
    },
    "Grape Healthy": {
        "crop": "Grape",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Train vines properly and perform annual dormant pruning.",
            "Maintain balanced nutrition with moderate nitrogen application.",
            "Ensure good drainage and avoid waterlogged soils.",
            "Scout canopy regularly for early signs of pest or disease.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # PEACH
    # -------------------------------------------------------------------------
    "Peach Healthy": {
        "crop": "Peach",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Apply dormant-season fungicide (copper or lime-sulfur) to reduce pathogen load.",
            "Thin fruit to improve air circulation and reduce brown rot risk.",
            "Prune to an open centre to maximise sunlight and airflow.",
            "Monitor for peach leaf curl and oriental fruit moth early in the season.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # POTATO
    # -------------------------------------------------------------------------
    "Potato Early Blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "is_healthy": False,
        "symptoms": (
            "Dark brown to black concentric-ringed (target-like) lesions appear on "
            "older, lower leaves first. Lesions are often surrounded by a yellow halo. "
            "Severe infections cause extensive defoliation and reduced tuber size."
        ),
        "causes": (
            "Caused by the fungus Alternaria solani. The pathogen overwinters in "
            "infected plant debris and soil. Warm temperatures (24-29 °C), alternating "
            "wet and dry periods, and nutrient-stressed plants are most susceptible."
        ),
        "prevention": [
            "Practice 2-3 year crop rotation with non-solanaceous crops.",
            "Use certified disease-free seed potatoes.",
            "Maintain adequate nitrogen and phosphorus fertility to reduce plant stress.",
            "Remove and destroy infected plant debris after harvest.",
            "Irrigate in the morning so foliage dries quickly during the day.",
        ],
        "chemical_treatment": [
            "Chlorothalonil (Bravo) at 2.0 g/L applied at 7-10 day intervals.",
            "Azoxystrobin (Quadris) at 0.6-0.8 mL/L for systemic protection.",
            "Mancozeb 75 WP at 2.5 g/L as a broad-spectrum protectant.",
        ],
        "organic_treatment": [
            "Copper-based fungicides (copper hydroxide) applied preventively.",
            "Bacillus subtilis-based biofungicides (Serenade) at 7-day intervals.",
            "Compost amendments to improve soil health and suppress soilborne inoculum.",
        ],
        "severity": "Medium",
    },
    "Potato Late Blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "is_healthy": False,
        "symptoms": (
            "Large, water-soaked, pale-green to dark-brown lesions appear on leaf tips "
            "and margins, often with a white-gray fuzzy mould on the underside. Lesions "
            "expand rapidly in wet weather, turning entire leaves black. Tubers develop "
            "firm, reddish-brown, granular rot."
        ),
        "causes": (
            "Caused by the oomycete Phytophthora infestans. The pathogen spreads via "
            "wind-borne sporangia and can devastate entire fields within days under "
            "cool, wet conditions (12-18 °C with sustained leaf wetness). Infected "
            "seed tubers and volunteer plants serve as primary inoculum sources."
        ),
        "prevention": [
            "Plant certified disease-free seed tubers.",
            "Destroy volunteer potato and tomato plants that harbour the pathogen.",
            "Apply preventive fungicides before forecast wet periods.",
            "Improve field drainage and avoid overhead irrigation.",
            "Hill soil around plant bases to protect tubers from spore wash-down.",
        ],
        "chemical_treatment": [
            "Metalaxyl-M + Mancozeb (Ridomil Gold MZ) at 2.5 g/L for curative and protectant action.",
            "Chlorothalonil (Bravo) at 2.0 g/L as a protectant applied on a 5-7 day schedule.",
            "Cymoxanil + Mancozeb (Curzate M) for systemic curative activity.",
        ],
        "organic_treatment": [
            "Copper hydroxide or Bordeaux mixture applied preventively at 5-7 day intervals.",
            "Bacillus amyloliquefaciens-based biocontrol agents to suppress sporangia germination.",
            "Remove and destroy all infected plant tissue immediately upon detection.",
        ],
        "severity": "High",
    },
    # -------------------------------------------------------------------------
    # RASPBERRY
    # -------------------------------------------------------------------------
    "Raspberry Healthy": {
        "crop": "Raspberry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Prune out spent floricanes immediately after harvest.",
            "Maintain narrow rows (30-45 cm) with good air movement.",
            "Provide adequate drip irrigation and avoid wetting canes.",
            "Apply mulch to suppress weeds and conserve soil moisture.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # SOYBEAN
    # -------------------------------------------------------------------------
    "Soybean Healthy": {
        "crop": "Soybean",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Inoculate seed with Bradyrhizobium japonicum for optimal nitrogen fixation.",
            "Rotate with corn or small grains to break pest and disease cycles.",
            "Scout fields regularly for aphids, bean leaf beetle, and other pests.",
            "Maintain proper soil pH (6.0-7.0) and fertility levels.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # SQUASH
    # -------------------------------------------------------------------------
    "Squash Powdery Mildew": {
        "crop": "Squash",
        "disease": "Powdery Mildew",
        "is_healthy": False,
        "symptoms": (
            "White, powdery, talc-like fungal colonies appear on upper and lower leaf "
            "surfaces, petioles, and stems. Infected leaves turn yellow, become brittle, "
            "and senesce prematurely. Severe infection reduces fruit size and quality."
        ),
        "causes": (
            "Caused primarily by the fungi Podosphaera xanthii and Erysiphe "
            "cichoracearum. Unlike most fungal diseases, powdery mildew thrives in "
            "warm, dry conditions (20-27 °C) with moderate humidity. Shade and poor "
            "air circulation exacerbate the disease."
        ),
        "prevention": [
            "Plant powdery-mildew-resistant squash varieties.",
            "Space plants adequately for good air circulation.",
            "Avoid excessive nitrogen fertilisation which promotes succulent growth.",
            "Water at the base of plants in the morning.",
            "Remove and destroy severely infected leaves promptly.",
        ],
        "chemical_treatment": [
            "Myclobutanil (Rally) applied at first sign of white colonies.",
            "Potassium bicarbonate (Kaligreen) at 2.5-3.0 g/L as a curative contact spray.",
            "Chlorothalonil (Bravo) at 2.0 g/L as a preventive protectant.",
        ],
        "organic_treatment": [
            "Potassium bicarbonate (baking soda alternative) at 5 g/L + surfactant.",
            "Neem oil (0.5-1 %) sprayed at 7-10 day intervals.",
            "Milk spray (40 % milk to water ratio) applied weekly as a foliar treatment.",
        ],
        "severity": "Medium",
    },
    # -------------------------------------------------------------------------
    # STRAWBERRY
    # -------------------------------------------------------------------------
    "Strawberry Healthy": {
        "crop": "Strawberry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Use certified disease-free transplants from reputable nurseries.",
            "Mulch with straw to reduce soil splash and fruit rot.",
            "Renovate matted-row beds annually to remove old foliage and thin plants.",
            "Provide drip irrigation to keep foliage dry.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
    # -------------------------------------------------------------------------
    # TOMATO
    # -------------------------------------------------------------------------
    "Tomato Bacterial Spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "symptoms": (
            "Small, dark, water-soaked spots with narrow yellow halos appear on leaves. "
            "Spots may coalesce, causing leaves to turn brown and drop. Fruit develop "
            "raised, scabby, dark lesions that crack and reduce market quality."
        ),
        "causes": (
            "Caused by Xanthomonas spp. (X. vesicatoria, X. euvesicatoria, "
            "X. gardneri, X. perforans). Bacteria spread via contaminated seed, "
            "rain splash, overhead irrigation, and worker handling. Warm, wet weather "
            "(24-30 °C) accelerates epidemics."
        ),
        "prevention": [
            "Use pathogen-free seed treated with hot water (50 °C for 25 minutes).",
            "Avoid overhead irrigation; use drip lines.",
            "Rotate with non-solanaceous crops for at least 2 years.",
            "Disinfect stakes, cages, and tools between seasons.",
            "Remove symptomatic plants and debris promptly.",
        ],
        "chemical_treatment": [
            "Copper hydroxide (Kocide 3000) + Mancozeb applied at 5-7 day intervals.",
            "Acibenzolar-S-methyl (Actigard) to induce systemic acquired resistance.",
            "Streptomycin (where legally permitted) at transplanting for early protection.",
        ],
        "organic_treatment": [
            "Fixed copper sprays (copper octanoate) at 5-7 day intervals.",
            "Bacillus subtilis-based biofungicides (Serenade ASO) applied as foliar spray.",
            "Hydrogen peroxide-based sanitisers for greenhouse surfaces and tools.",
        ],
        "severity": "High",
    },
    "Tomato Early Blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "is_healthy": False,
        "symptoms": (
            "Dark brown, concentric-ringed (target-board) lesions appear on older, "
            "lower leaves. A yellow halo often surrounds each lesion. Progressive "
            "defoliation exposes fruit to sunscald and reduces yield."
        ),
        "causes": (
            "Caused by the fungus Alternaria solani. The pathogen persists in soil "
            "and on infected plant debris. Warm temperatures (24-29 °C), high humidity, "
            "and nutrient-deficient or stressed plants favour rapid infection."
        ),
        "prevention": [
            "Rotate crops with non-solanaceous plants for 2-3 years.",
            "Stake or cage plants to keep foliage off the ground.",
            "Mulch around the base to prevent soil splash onto lower leaves.",
            "Remove infected lower leaves at first sign of symptoms.",
            "Ensure adequate nitrogen, phosphorus, and potassium nutrition.",
        ],
        "chemical_treatment": [
            "Chlorothalonil (Bravo) at 2.0 g/L on a 7-10 day schedule.",
            "Azoxystrobin (Quadris) at 0.6-0.8 mL/L for systemic protection.",
            "Mancozeb 75 WP at 2.5 g/L as a protectant during vegetative growth.",
        ],
        "organic_treatment": [
            "Copper-based fungicides (Bordeaux mixture) applied at 7-day intervals.",
            "Bacillus subtilis (Serenade) applied preventively as a foliar spray.",
            "Compost tea applications to support foliar microbiome and suppress pathogens.",
        ],
        "severity": "Medium",
    },
    "Tomato Late Blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "is_healthy": False,
        "symptoms": (
            "Large, irregular, water-soaked, dark-green to brown lesions appear on "
            "leaves and stems, often with a white, fuzzy mould on the underside during "
            "humid conditions. Entire plants can collapse within days. Fruit develop "
            "firm, greasy, brown blotches."
        ),
        "causes": (
            "Caused by the oomycete Phytophthora infestans. The same pathogen that "
            "caused the Irish Potato Famine. Sporangia are wind-dispersed over long "
            "distances. Cool, wet conditions (12-18 °C with prolonged leaf wetness) "
            "are ideal for rapid epidemic development."
        ),
        "prevention": [
            "Plant late-blight-resistant or tolerant cultivars (e.g., Defiant, Mountain Magic).",
            "Destroy volunteer tomato and potato plants.",
            "Apply protectant fungicides before forecast rainy periods.",
            "Improve air circulation through proper spacing and pruning.",
            "Destroy all infected plant material immediately - do not compost.",
        ],
        "chemical_treatment": [
            "Metalaxyl-M + Mancozeb (Ridomil Gold MZ) at 2.5 g/L for curative action.",
            "Chlorothalonil (Bravo) at 2.0 g/L applied on a 5-7 day preventive schedule.",
            "Cymoxanil + Famoxadone (Tanos) for systemic and protectant activity.",
        ],
        "organic_treatment": [
            "Copper hydroxide or Bordeaux mixture applied at 5-7 day intervals.",
            "Remove and destroy all infected tissue immediately upon detection.",
            "Serenade (Bacillus subtilis) as a supplemental biocontrol spray.",
        ],
        "severity": "High",
    },
    "Tomato Leaf Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "is_healthy": False,
        "symptoms": (
            "Pale green to yellowish diffuse spots appear on upper leaf surfaces. "
            "The corresponding undersides develop an olive-green to grayish-brown "
            "velvety mould. Severely affected leaves curl, wither, and drop."
        ),
        "causes": (
            "Caused by the fungus Passalora fulva (syn. Cladosporium fulvum, "
            "Fulvia fulva). The pathogen thrives in high humidity (>85 %) and moderate "
            "temperatures (22-25 °C), especially in poorly ventilated greenhouses. "
            "Spores are easily spread by air currents and on workers' clothing."
        ),
        "prevention": [
            "Increase greenhouse ventilation and reduce relative humidity below 85 %.",
            "Space plants adequately and prune lower leaves for air movement.",
            "Water at the base of plants and avoid wetting foliage.",
            "Use resistant cultivars carrying Cf resistance genes (Cf-2, Cf-5, Cf-9).",
        ],
        "chemical_treatment": [
            "Chlorothalonil (Bravo) at 2.0 g/L on a 7-10 day schedule.",
            "Azoxystrobin (Quadris) at 0.6-0.8 mL/L for systemic activity.",
            "Mancozeb 75 WP at 2.0 g/L as a protectant in greenhouse environments.",
        ],
        "organic_treatment": [
            "Potassium bicarbonate at 5 g/L + surfactant applied to foliage.",
            "Neem oil (0.5-1 %) sprayed at 7-day intervals.",
            "Improve ventilation as the primary cultural control measure.",
        ],
        "severity": "Medium",
    },
    "Tomato Septoria Leaf Spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "is_healthy": False,
        "symptoms": (
            "Numerous small (2-3 mm), circular spots with dark brown margins and gray "
            "to tan centres appear on lower leaves. Tiny dark pycnidia (fruiting bodies) "
            "are visible in the spot centres with a hand lens. Heavy infection causes "
            "extensive defoliation from the bottom up."
        ),
        "causes": (
            "Caused by the fungus Septoria lycopersici. The pathogen overwinters in "
            "infected plant debris and on solanaceous weeds. Splashing water from rain "
            "or irrigation spreads spores upward through the canopy. Warm, wet weather "
            "(20-25 °C) promotes rapid disease spread."
        ),
        "prevention": [
            "Mulch around plants to reduce soil splash onto lower foliage.",
            "Stake or cage plants and prune lower branches to keep foliage off the ground.",
            "Rotate away from tomatoes and other solanaceous crops for 2+ years.",
            "Remove and destroy all tomato debris at season end.",
            "Avoid overhead irrigation; use drip systems.",
        ],
        "chemical_treatment": [
            "Chlorothalonil (Bravo) at 2.0 g/L starting at first symptom appearance.",
            "Mancozeb 75 WP at 2.5 g/L applied at 7-10 day intervals.",
            "Azoxystrobin (Quadris) at 0.6-0.8 mL/L for systemic and protectant control.",
        ],
        "organic_treatment": [
            "Copper-based fungicides (copper hydroxide) applied at 7-day intervals.",
            "Bacillus subtilis (Serenade) as a preventive foliar biofungicide.",
            "Remove infected lower leaves immediately to slow disease progression.",
        ],
        "severity": "Medium",
    },
    "Tomato Mosaic Virus": {
        "crop": "Tomato",
        "disease": "Mosaic Virus",
        "is_healthy": False,
        "symptoms": (
            "Leaves show a mottled mosaic pattern of light and dark green areas, often "
            "with distortion, curling, and fern-like narrowing of leaflets. Plants may "
            "be stunted with reduced fruit set. Fruit can develop internal browning and "
            "uneven ripening."
        ),
        "causes": (
            "Caused by Tomato mosaic virus (ToMV), a tobamovirus closely related to "
            "Tobacco mosaic virus (TMV). The virus is extremely stable, transmitted "
            "mechanically through contaminated hands, tools, and infected seed. It "
            "persists on dry surfaces and in soil for months to years."
        ),
        "prevention": [
            "Use certified virus-free seed and resistant cultivars carrying Tm-2² gene.",
            "Wash hands thoroughly with soap and milk (protein denatures the virus) before handling plants.",
            "Disinfect all tools with 10 % trisodium phosphate or dilute bleach.",
            "Remove and destroy infected plants immediately to prevent spread.",
            "Avoid tobacco use near tomato plants (TMV source).",
        ],
        "chemical_treatment": [
            "No chemical cure exists for viral infections once established.",
            "Insecticidal soap to control secondary vector insects if applicable.",
            "Focus entirely on prevention and sanitation.",
        ],
        "organic_treatment": [
            "Milk sprays (20 % milk solution) applied to tools and hands as a protein-based disinfectant.",
            "Plant resistant cultivars as the most effective long-term strategy.",
            "Maintain strict greenhouse hygiene with foot baths and tool disinfection.",
        ],
        "severity": "High",
    },
    "Tomato Yellow Leaf Curl Virus": {
        "crop": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "is_healthy": False,
        "symptoms": (
            "Young leaves curl upward and inward, becoming cupped and thickened with "
            "prominent yellowing of leaf margins. Plants are severely stunted with "
            "bushy, erect growth. Flower drop is common and fruit production is "
            "drastically reduced or eliminated."
        ),
        "causes": (
            "Caused by Tomato yellow leaf curl virus (TYLCV), a begomovirus. The virus "
            "is transmitted exclusively by the silverleaf whitefly (Bemisia tabaci) in "
            "a persistent, circulative manner. High whitefly populations and warm "
            "conditions (25-30 °C) drive rapid epidemic spread."
        ),
        "prevention": [
            "Use TYLCV-resistant cultivars carrying Ty-1, Ty-2, or Ty-3 resistance genes.",
            "Install UV-reflective mulch to repel whiteflies from crop rows.",
            "Use fine-mesh insect-exclusion netting (50-mesh) in greenhouses.",
            "Implement whitefly monitoring with yellow sticky traps.",
            "Remove and destroy infected plants and nearby weed hosts immediately.",
        ],
        "chemical_treatment": [
            "Imidacloprid (Admire) applied as a soil drench at transplanting for whitefly control.",
            "Pyriproxyfen (Knack) to disrupt whitefly reproduction and reduce virus spread.",
            "Cyantraniliprole (Verimark) as a systemic drench for extended whitefly suppression.",
        ],
        "organic_treatment": [
            "Neem oil (0.5-1 %) or neem-based insecticides to deter whitefly feeding.",
            "Release of Encarsia formosa or Eretmocerus eremicus parasitoid wasps for biological control.",
            "Insecticidal soap applied at 2-3 day intervals to knock down whitefly populations.",
        ],
        "severity": "High",
    },
    "Tomato Healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": (
            "No disease symptoms. The leaf appears healthy with normal coloration "
            "and texture."
        ),
        "causes": "N/A",
        "prevention": [
            "Rotate tomatoes with non-solanaceous crops every 2-3 years.",
            "Stake or cage plants for support and improved air circulation.",
            "Apply consistent, even watering with drip irrigation.",
            "Monitor for pests and diseases weekly throughout the season.",
        ],
        "chemical_treatment": [
            "No chemical treatment required for healthy plants.",
        ],
        "organic_treatment": [
            "No organic treatment required for healthy plants.",
        ],
        "severity": "Low",
    },
}


def get_disease_info(class_name: str) -> dict:
    """Look up disease information for a given unified class name.

    Parameters
    ----------
    class_name : str
        One of the 27 unified PlantDoc class names (e.g. ``"Tomato Early Blight"``).

    Returns
    -------
    dict
        A dictionary containing crop, disease, symptoms, causes, prevention tips,
        chemical and organic treatment recommendations, and severity rating.
        If *class_name* is not found, a default ``"Unknown"`` entry is returned.
    """
    return DISEASE_DB.get(
        class_name,
        {
            "crop": "Unknown",
            "disease": "Unknown",
            "is_healthy": False,
            "symptoms": "No information available for this class.",
            "causes": "Unknown",
            "prevention": ["Consult a local agricultural extension service."],
            "chemical_treatment": ["Consult a licensed agronomist for recommendations."],
            "organic_treatment": ["Consult a local agricultural extension service."],
            "severity": "Unknown",
        },
    )


if __name__ == "__main__":
    print(f"{'Class Name':<40} {'Severity':<10} {'Healthy?'}")
    print("-" * 62)
    for name, info in DISEASE_DB.items():
        healthy_marker = "Yes" if info["is_healthy"] else ""
        print(f"{name:<40} {info['severity']:<10} {healthy_marker}")
    print(f"\nTotal classes: {len(DISEASE_DB)}")
