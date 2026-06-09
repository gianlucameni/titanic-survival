import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io

class Graphs:
    def __init__(self, df):
        self.df = df

    def survival_by_feature(self):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Survival Distribution by Feature', fontsize=16, fontweight='bold')

        survival_counts = self.df['Survived'].value_counts()
        axes[0, 0].bar(['Did Not Survive', 'Survived'], survival_counts.values,
                       color=['#e74c3c', '#2ecc71'])
        axes[0, 0].set_title('Overall Survival Count')
        for i, v in enumerate(survival_counts.values):
            axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

        sex_survival = self.df.groupby('Sex')['Survived'].mean()
        axes[0, 1].bar(['Male', 'Female'], sex_survival.values,
                       color=['#3498db', '#e91e8c'])
        axes[0, 1].set_title('Survival Rate by Gender')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(sex_survival.values):
            axes[0, 1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

        pclass_survival = self.df.groupby('Pclass')['Survived'].mean()
        axes[0, 2].bar(['1st Class', '2nd Class', '3rd Class'], pclass_survival.values,
                       color=['#f39c12', '#95a5a6', '#7f8c8d'])
        axes[0, 2].set_title('Survival Rate by Class')
        axes[0, 2].set_ylim(0, 1)
        for i, v in enumerate(pclass_survival.values):
            axes[0, 2].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

        axes[1, 0].hist(self.df[self.df['Survived'] == 1]['Age'].dropna(), bins=20,
                        alpha=0.7, color='#2ecc71', label='Survived')
        axes[1, 0].hist(self.df[self.df['Survived'] == 0]['Age'].dropna(), bins=20,
                        alpha=0.7, color='#e74c3c', label='Did Not Survive')
        axes[1, 0].set_title('Age Distribution')
        axes[1, 0].legend()

        axes[1, 1].hist(self.df[self.df['Survived'] == 1]['Fare'], bins=20,
                        alpha=0.7, color='#2ecc71', label='Survived')
        axes[1, 1].hist(self.df[self.df['Survived'] == 0]['Fare'], bins=20,
                        alpha=0.7, color='#e74c3c', label='Did Not Survive')
        axes[1, 1].set_title('Fare Distribution')
        axes[1, 1].legend()

        sns.countplot(data=self.df, x='Sex', hue='Survived', palette=['#e74c3c', '#2ecc71'], ax=axes[1, 2])
        axes[1, 2].set_title('Survival Count by Gender')
        axes[1, 2].legend(title='Survived', labels=['No', 'Yes'])


        plt.tight_layout()
        plt.show()

    def plot_distribution(self, column):
        sns.histplot(self.df[column], kde=True)
        plt.show()

    def plot_outliers(self, column):
        sns.boxplot(self.df[column])
        plt.show()


    def correlation_matrix(self):
        # selezione colonne numeriche
        num_df = self.df.select_dtypes(include=['number'])
        # matrice di correlazione
        corr = num_df.corr()
        # maschera triangolo superiore
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
        )

        plt.title("Correlation Heatmap (Lower Triangle)")
        plt.show()
        #buf = io.BytesIO()
        #plt.savefig(buf, format="png", bbox_inches="tight")
        #buf.seek(0)

        #img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        #plt.close()

        #return img_base64
