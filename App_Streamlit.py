import streamlit as st
import pandas as pd
from Chargement_de_modele import predire_defaut , charger_modeles

st.title("Application D'evaluation du risque de Defaut de paiement")
nouveaux_noms = [
    "age_emprunteur", 
    "revenu_annuel", 
    "anciennete_pro", 
    "montant_credit", 
    "taux_interet", 
    "statut_pret", 
    "antecedent_defaut"
]
with st.form("formulaire_scoring"):
    st.subheader("Information de l'emprunteur")

    age_emprunteur = st.number_input("Âge de l'emprunteur ! ",min_value = 18 , max_value=100)
    revenu_annuel =st.number_input("Entrez votre revenu_annuel !", min_value = 0 )
    anciennete_pro = st.number_input("Depuis combien de temps travailler vous ?")
    montant_credit = st.number_input("Montant du credit",min_value = 0)
    taux_interet =st.number_input("Quel est le taux d'interet",min_value=0.0)
    antecedent_defaut= st.selectbox("Avez vous des antecedent de defaut de paiement?",options =["Oui","Non"])
    submit = st.form_submit_button("Valider")

if submit:
    antecedent_num = 1 if antecedent_defaut == "Oui" else 0

    donnees_clients = pd.DataFrame([
          {
              "age_emprunteur": age_emprunteur,
              "revenu_annuel": revenu_annuel,
              "anciennete_pro": anciennete_pro,
              "montant_credit": montant_credit,
              "taux_interet": taux_interet,
              "antecedent_defaut": antecedent_num,
          }
      ]
  )

try:
    proba_defaut = predire_defaut(donnees_clients)

    
    st.subheader("Résultat de l'analyse")
    st.metric(label="Probabilité de défaut", value=f"{proba_defaut:.1%}")

    if proba_defaut >= 0.40:
      st.error("Risque élevé : le dossier présente une forte probabilité d'impayé.")
    elif proba_defaut >= 0.20:
      st.warning("Risque modéré : analyse manuelle recommandée.")
    else:
      st.success("Risque faible : dossier favorable.")

except Exception as e:
    st.error(f"Erreur lors du calcul : {e}")