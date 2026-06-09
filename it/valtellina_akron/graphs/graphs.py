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

        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        family_survival = self.df.groupby('FamilySize')['Survived'].mean()
        axes[1, 2].bar(family_survival.index, family_survival.values, color='#9b59b6')
        axes[1, 2].set_title('Survival by Family Size')
        axes[1, 2].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    def plot_distribution(self, column):
        sns.histplot(self.df[column], kde=True)
        plt.show()

    def plot_outliers(self, column):
        sns.boxplot(self.df[column])
        plt.show()
