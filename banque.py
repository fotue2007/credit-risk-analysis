# projet de machine learning 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve, auc
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
# etude des donnees du dataset credit_risk_dataset.csv  
# creation du dataframme 
df = pd.read_csv("credit_risk_dataset.csv")
            # 5 premiere ligne du tableau 
print(df.head())
            # etude du type des colonnes colonne 
print(df.info())
            # recuperation des colone qu'on va utiliser 
df = df [["person_age","person_income" ,"person_emp_length" ,"loan_amnt" ,"loan_int_rate","loan_status" ,"cb_person_default_on_file"]]
nouveaux_noms = [
    "age_emprunteur", 
    "revenu_annuel", 
    "anciennete_pro", 
    "montant_credit", 
    "taux_interet", 
    "statut_pret", 
    "antecedent_defaut"
]
print(df.describe())
            # changement du type des colonnes
df["cb_person_default_on_file"] = df["cb_person_default_on_file"].astype('string')
print(df.info())
            # verification de l'existance des valeur manquante
print(df.isna().any())  # manque de valeur dans le taux d'interet et dans l'anciennete dans le travail 
            # calcul des valeur moyenne pour les colone avec donnee manquante 
taux_moyen = df["loan_int_rate"].mean()
anciennete_moyenne = df["person_emp_length"].mean()
            # remplacement des valeur manqunte par la moyenne des valeur
df["loan_int_rate"] = df["loan_int_rate"].fillna(taux_moyen)
df["person_emp_length"] = df["person_emp_length"].fillna(anciennete_moyenne)
            #verification de l'application des changement
print(df.isna().any()) 
            # remplacement de la colonne cb_person_default_file en binaire (0 s'il n'a jamis fait de defaut de paiement et 1 dans le cas contraire)
df["cb_person_default_on_file"] = [0 if n.lower()=="n" else 1 for n in df["cb_person_default_on_file"]]
            #verification des changements
print(df.head())
print(df.info())
            # mofication des noms des colonnes
df.columns = [col for col in nouveaux_noms]
print(df.info()) 
        # a present des donnnees toute propre prete a etre utiliser
moyenner = df.groupby("statut_pret")["revenu_annuel"].mean()
print(moyenner)
moyenneT = df.groupby("statut_pret")["taux_interet"].mean()
print(moyenneT)
#detection et suppression des doublons
print(df.duplicated())
df = df.drop_duplicates()
#Representation des donnees 
# --- 1. LE CAMEMBERT (STATUT DU PRÊT) ---
df["statut_pret"].value_counts().plot(kind="pie" , autopct= "%1.1f%%" )
plt.title("Répartition des Classes (Variable Cible : statut_pret)")
plt.show()

# --- 2. RÉPARTITION DE L'ANCIENNETÉ (HISTOGRAMME) ---
plt.figure(figsize=(10, 6))
sns.histplot(df["anciennete_pro"], bins=20, kde=True, color="skyblue")
plt.title("Distribution de l'ancienneté professionnelle")
plt.xlabel("Années d'ancienneté")
plt.ylabel("Nombre d'emprunteurs")
plt.show()


# --- GRAPHIQUE 1 : Matrice de Corrélation ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Corrélation entre les variables de crédit')
plt.savefig('correlation_matrix.png')
plt.show()

# --- GRAPHIQUE 2 : Répartition du revenu selon le statut du prêt ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='statut_pret', y='revenu_annuel', data=df)
plt.title('Distribution des Revenus par Statut de Prêt')
plt.savefig('revenu_vs_statut.png')
plt.show()

# --- GRAPHIQUE 3 : Taux d'intérêt et défaut ---
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x="taux_interet", hue="statut_pret", fill=True)
plt.title('Impact du Taux d\'intérêt sur le Risque de Défaut')
plt.savefig('taux_distribution.png')
plt.show()

# --- GRAPHIQUE 4 : Antécédents de défaut ---
plt.figure(figsize=(8, 6))
sns.countplot(x='antecedent_defaut', hue='statut_pret', data=df)
plt.title('Influence des Antécédents sur le Statut Actuel')
plt.savefig('antecedents_count.png') 
plt.show()   


#machine learning 
#1- division des donnees en respectant les proportion du data set 
X=df[[ "age_emprunteur", "revenu_annuel",  "anciennete_pro", "montant_credit", "taux_interet", "antecedent_defaut"]]
y=df["statut_pret"]
X_train ,X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,stratify=y,random_state=42)

#2-instantiation du modele de regression 
model_credit = LogisticRegression()

model_credit.fit(X_train,y_train)
#3-Prediction
y_pred = model_credit.predict(X_test)
y_prob = model_credit.predict_proba(X_test)[:,1]

# --- 5. ÉVALUATION (Rapport de classification) ---
print("Rapport de Classification :")
print(classification_report(y_test, y_pred))

# --- 6. GRAPHIQUE : MATRICE DE CONFUSION ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Remboursé (0)', 'Défaut (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Matrice de Confusion")
plt.show()

# --- 7. GRAPHIQUE : COURBE ROC (Performance globale) ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.show()