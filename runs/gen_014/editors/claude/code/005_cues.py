# Hypothesis: (labels) hand-labelled decisive stem cues (verbatim substrings) for finished-wrong verbal questions, plus a non-decisive stem sentence as control.
import json
from wsp.editor import *
Q = questions()
CUES = {
 "mmlupro_1171": "When burger joint owner loses customers and business income to pizza place owner, can he sue for conversion?",
 "mmlupro_2654": "he said he had dreamed he was falling to his death",
 "mmlupro_6110": "All patients enrolled in the study are informed that they would be treated with the tyrosine kinase inhibitor.",
 "mmlupro_6562": "who has smoked 20 cigarettes daily for the past 38 years",
 "mmlupro_6800": "There is tenderness to palpation over the left cheek and no transillumination over the left maxillary sinus.",
 "mmlupro_1196": "the court extended the TRO for another 30 days",
 "mmlupro_1643": "Two months after the court entered judgment, the defendant was given a videotape made that day showing the plaintiff jogging with her doctor.",
 "mmlupro_1705": "Without attempting to get a warrant, the police stopped all the cars waiting for the ferry.",
 "mmlupro_6614": "BMI is 37 kg/m2",
 "mmlupro_6829": "If the world price of copper exceeds the domestic (U.S.) price of copper",
 "mmlupro_1445": "The statute does not, however, restrict the sale of red meat products.",
 "mmlupro_1275": "without express permission of the mayor",
 "mmlupro_1409": "which of the following is the WEAKEST defense for the city?",
 "mmlupro_1754": "I will lend him $50,000 at 10 percent interest provided he signs a contract with a builder for construction of the shopping center on this tract.",
 "mmlupro_5510": "which of the following is the odd one out?",
}
# non-decisive stem part of similar length (verbatim), or None -> neutral filler
CTRL = {
 "mmlupro_1171": "An owner of a pizza and Italian foods restaurant opens a new location on a street where another owner has a burger restaurant.",
 "mmlupro_2654": "When his wife asked him what had happened",
 "mmlupro_6110": "Treatment efficacy is determined based on the results of complete blood counts and bone marrow assessments conducted regularly throughout the study.",
 "mmlupro_6562": "Three days later, she has a marked increase",
 "mmlupro_6800": "She does not smoke. Her husband and children have had no recent illness.",
 "mmlupro_1196": "Two days before his home was to be sold at a foreclosure sale",
 "mmlupro_1643": "A plaintiff sued a defendant in federal court for injuries arising out of an accident involving the parties.",
 "mmlupro_1705": "The only way onto or off the island where he lived was a ferry.",
 "mmlupro_6614": "She walks 3 miles daily",
 "mmlupro_6829": None,
 "mmlupro_1445": "Producers of red meat have challenged the statute as a violation of their free speech rights",
 "mmlupro_1275": "After a few days of picketing",
 "mmlupro_1409": "For the past 20 years a city by the beach has been a popular location for surfboarding.",
 "mmlupro_1754": "After an extensive search, he purchased a three-acre undeveloped tract of land for the purpose of developing a small shopping center.",
 "mmlupro_5510": None,
}
if __name__ == "__main__":
    for q, c in CUES.items():
        st = Q[q]["question"]
        ok = c in st; ok2 = CTRL[q] is None or CTRL[q] in st
        print(q, "cue in stem:", ok, "ctrl in stem:", ok2, "cue toks", len(tok.encode(c).ids if hasattr(tok.encode(c),'ids') else tok.encode(c)))
