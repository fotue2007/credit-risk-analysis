import joblib
import pandas as pd

print(" Chargement du modèle et du scaler...")
model = joblib.load("model_credit.joblib")
scaler = joblib.load("scaler_credit.joblib")
print("✅ Modèle chargé avec succès !\n")

try:
    age_emprunteur = int(input("👤 Âge de l'emprunteur (ex: 30) : "))
    revenu_annuel = float(input("💰 Revenu annuel en $ (ex: 50000) : "))
    anciennete_pro = float(input("💼 Ancienneté professionnelle en années (ex: 5) : "))
    montant_credit = float(input("💵 Montant du crédit demandé en $ (ex: 10000) : "))
    taux_interet = float(input("📊 Taux d'intérêt proposé en % (ex: 10.5) : "))

    antecedent_input = input("⚠️  Antécédent de défaut de paiement ? (oui/non) : ").strip().lower()
    antecedent_defaut = 1 if antecedent_input in ["oui", "o", "yes", "y", "1"] else 0

except ValueError:
    print("\n❌ Erreur : veuillez entrer des valeurs numériques valides.")
    exit()

donnees = pd.DataFrame([[
    age_emprunteur,
    revenu_annuel,
    anciennete_pro,
    montant_credit,
    taux_interet,
    antecedent_defaut
]], columns=[
    "age_emprunteur", "revenu_annuel", "anciennete_pro",
    "montant_credit", "taux_interet", "antecedent_defaut"
])

donnees_scaled = scaler.transform(donnees)

prediction = model.predict(donnees_scaled)[0]
probabilite_defaut = model.predict_proba(donnees_scaled)[0][1]

df_coef = pd.DataFrame({
    "Variable": feature_names,
    "Coefficient": coefficients,
    "Impact": ["🔴 Augmente risque" if c > 0 else "🟢 Diminue risque" for c in coefficients]
})

# 5. Trier par importance (valeur absolue)
df_coef["Importance"] = df_coef["Coefficient"].abs()
df_coef = df_coef.sort_values("Importance", ascending=False)


# 6. Afficher
print("\n" + "="*70)
print("📊 IMPACT DES VARIABLES SUR LA PROBABILITÉ DE DÉFAUT")
print("="*70)
print(df_coef.to_string(index=False))
print("="*70)
print("\n" + "=" * 60)
print("  RÉSULTAT DE L'ÉVALUATION")
print("=" * 60)

print(f"\ Données saisies :")
print(f"   • Âge                : {age_emprunteur} ans")
print(f"   • Revenu annuel      : {revenu_annuel:,.0f} $")
print(f"   • Ancienneté pro     : {anciennete_pro} ans")
print(f"   • Montant du crédit  : {montant_credit:,.0f} $")
print(f"   • Taux d'intérêt     : {taux_interet} %")
print(f"   • Antécédent défaut  : {'Oui' if antecedent_defaut == 1 else 'Non'}")

print(f"\n🎯 Probabilité de défaut : {probabilite_defaut * 100:.2f} %")

if prediction == 0:
    print("\n✅ DÉCISION : Crédit recommandé (faible risque)")
else:
    print("\n⚠️  DÉCISION : Crédit à risque (fort risque de défaut)")