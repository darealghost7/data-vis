import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class AdultDataAnalyzer:
    def __init__(self, file_path):
        self.data = None
        self.file_path = file_path
        self.column_mapping = {}
        self.setup_plot_style()

    def setup_plot_style(self):
        """Configure consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        self.colors = {
            '<=50K': '#1f77b4',
            '>50K': '#ff7f0e',
            'female': '#9467bd',
            'male': '#2ca02c'
        }

    def standardize_column_names(self):
        """Standardize column names to handle different naming conventions"""
        # Common variations of column names
        possible_names = {
            'hours_per_week': ['hours-per-week', 'hours_per_week', 'hours.per.week', 'hrs_per_week'],
            'education_num': ['education-num', 'education_num', 'education.num'],
            'marital_status': ['marital-status', 'marital_status', 'marital.status'],
            'capital_gain': ['capital-gain', 'capital_gain', 'capital.gain'],
            'capital_loss': ['capital-loss', 'capital_loss', 'capital.loss'],
            'native_country': ['native-country', 'native_country', 'native.country']
        }

        # Create mapping from current names to standardized names
        current_columns = self.data.columns.tolist()
        self.column_mapping = {}

        for standardized, variants in possible_names.items():
            for variant in variants:
                if variant in current_columns:
                    self.column_mapping[standardized] = variant
                    break

        # Keep original names for columns not in our mapping
        for col in current_columns:
            if col not in [v for variants in possible_names.values() for v in variants]:
                self.column_mapping[col] = col

    def get_column_name(self, standard_name):
        """Get the actual column name from standardized name"""
        return self.column_mapping.get(standard_name, standard_name)

    def load_and_clean_data(self):
        """Load dataset and perform data cleaning"""
        try:
            # Load dataset
            self.data = pd.read_csv(self.file_path)

            print("Original columns in dataset:")
            print(self.data.columns.tolist())
            print("\n" + "=" * 50 + "\n")

            # Standardize column names
            self.standardize_column_names()

            print("Dataset Overview:")
            print(self.data.head())
            print("\n" + "=" * 50 + "\n")

            # Data cleaning
            self._handle_missing_values()
            self._remove_duplicates()

            print("Cleaned Dataset Info:")
            print(self.data.info())
            print("\n" + "=" * 50 + "\n")

        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _handle_missing_values(self):
        """Handle missing values represented by '?'"""
        # Count missing values before cleaning
        missing_before = (self.data == '?').sum().sum()
        print(f"Missing values ('?') found: {missing_before}")

        # Replace '?' with NaN
        self.data.replace('?', pd.NA, inplace=True)

        # Impute categorical variables with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isna().any():
                mode_value = self.data[col].mode()[0]
                self.data[col].fillna(mode_value, inplace=True)
                print(f"Imputed categorical column '{col}' with mode: {mode_value}")

        # Impute numerical variables with median
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if self.data[col].isna().any():
                median_value = self.data[col].median()
                self.data[col].fillna(median_value, inplace=True)
                print(f"Imputed numerical column '{col}' with median: {median_value:.2f}")

    def _remove_duplicates(self):
        """Remove duplicate records"""
        duplicates_before = self.data.duplicated().sum()
        self.data.drop_duplicates(inplace=True)
        duplicates_after = self.data.duplicated().sum()
        print(f"Removed {duplicates_before - duplicates_after} duplicate records")

    def create_visualizations(self):
        """Create the 2x2 grid of visualizations"""
        try:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comprehensive Analysis of Adult Income Dataset',
                         fontsize=16, fontweight='bold', y=0.98)

            self._create_stacked_bar_chart(axs[0, 0])
            self._create_line_graph(axs[0, 1])
            self._create_histogram(axs[1, 0])
            self._create_grouped_bar_chart(axs[1, 1])

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            raise

    def _create_stacked_bar_chart(self, ax):
        """Create stacked bar chart for gender and income distribution"""
        try:
            gender_col = self.get_column_name('sex')
            income_col = self.get_column_name('income')

            income_gender = self.data.groupby([gender_col, income_col]).size().unstack()

            income_gender.plot(kind='bar', stacked=True, ax=ax,
                               color=[self.colors['<=50K'], self.colors['>50K']],
                               edgecolor='black', linewidth=0.5)

            ax.set_title('Income Distribution by Gender', fontsize=14, fontweight='bold')
            ax.set_xlabel('Gender', fontweight='bold')
            ax.set_ylabel('Number of Individuals', fontweight='bold')
            ax.legend(title='Income Category', title_fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Add value annotations on bars
            for container in ax.containers:
                ax.bar_label(container, label_type='center', fontsize=9,
                             color='white', fontweight='bold')

        except Exception as e:
            print(f"Error in stacked bar chart: {e}")
            ax.set_title('Error: Could not create chart', color='red')

    def _create_line_graph(self, ax):
        """Create line graph for age vs hours worked by income"""
        try:
            age_col = self.get_column_name('age')
            income_col = self.get_column_name('income')
            hours_col = self.get_column_name('hours_per_week')

            # Create age groups for better visualization
            self.data['age_group'] = pd.cut(self.data[age_col],
                                            bins=range(15, 100, 5),
                                            labels=[f'{i}-{i + 4}' for i in range(15, 95, 5)])

            age_hours = self.data.groupby(['age_group', income_col])[hours_col].mean().unstack()

            # Convert age groups to numeric for plotting
            x_positions = np.arange(len(age_hours))

            # Check if both income categories exist
            if '<=50K' in age_hours.columns:
                ax.plot(x_positions, age_hours['<=50K'],
                        color=self.colors['<=50K'], linewidth=2.5, marker='o', markersize=4,
                        label='<=50K')

            if '>50K' in age_hours.columns:
                ax.plot(x_positions, age_hours['>50K'],
                        color=self.colors['>50K'], linewidth=2.5, marker='s', markersize=4,
                        label='>50K')

            ax.set_title('Average Weekly Hours Worked by Age Group and Income',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Age Group (years)', fontweight='bold')
            ax.set_ylabel('Average Hours Worked Per Week', fontweight='bold')
            ax.legend(title='Income Bracket', title_fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set x-axis labels with rotation for better readability
            ax.set_xticks(x_positions)
            ax.set_xticklabels(age_hours.index, rotation=45, ha='right')

        except Exception as e:
            print(f"Error in line graph: {e}")
            ax.set_title('Error: Could not create chart', color='red')

    def _create_histogram(self, ax):
        """Create histogram for weekly work hours distribution"""
        try:
            income_col = self.get_column_name('income')
            hours_col = self.get_column_name('hours_per_week')

            low_income_data = self.data[self.data[income_col] == '<=50K']
            high_income_data = self.data[self.data[income_col] == '>50K']

            low_income_hours = low_income_data[hours_col] if not low_income_data.empty else []
            high_income_hours = high_income_data[hours_col] if not high_income_data.empty else []

            ax.hist(low_income_hours, bins=25, alpha=0.7, label='Income <=50K',
                    color=self.colors['<=50K'], edgecolor='black', linewidth=0.5)
            ax.hist(high_income_hours, bins=25, alpha=0.7, label='Income >50K',
                    color=self.colors['>50K'], edgecolor='black', linewidth=0.5)

            ax.set_title('Distribution of Weekly Work Hours by Income Group',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Hours Worked Per Week', fontweight='bold')
            ax.set_ylabel('Number of Individuals', fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add vertical line for typical 40-hour work week
            ax.axvline(x=40, color='red', linestyle='--', alpha=0.8,
                       label='Typical 40h Week')
            ax.legend()

        except Exception as e:
            print(f"Error in histogram: {e}")
            ax.set_title('Error: Could not create chart', color='red')

    def _create_grouped_bar_chart(self, ax):
        """Create grouped bar chart for education level by occupation"""
        try:
            occupation_col = self.get_column_name('occupation')
            income_col = self.get_column_name('income')
            education_col = self.get_column_name('education_num')

            education_occupation = self.data.groupby([occupation_col, income_col])[education_col].mean().unstack()

            # Sort by total education level for better visualization
            education_occupation['total'] = education_occupation.mean(axis=1)
            education_occupation = education_occupation.sort_values('total', ascending=False)
            education_occupation = education_occupation.drop('total', axis=1)

            x_positions = np.arange(len(education_occupation))
            bar_width = 0.35

            # Check if both income categories exist
            if '<=50K' in education_occupation.columns:
                ax.bar(x_positions - bar_width / 2, education_occupation['<=50K'],
                       bar_width, label='Income <=50K', color=self.colors['<=50K'],
                       edgecolor='black', linewidth=0.5)

            if '>50K' in education_occupation.columns:
                ax.bar(x_positions + bar_width / 2, education_occupation['>50K'],
                       bar_width, label='Income >50K', color=self.colors['>50K'],
                       edgecolor='black', linewidth=0.5)

            ax.set_title('Average Education Level by Occupation and Income Group',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Occupation', fontweight='bold')
            ax.set_ylabel('Average Education Level\n(higher number = more education)',
                          fontweight='bold')
            ax.legend(title='Income Group', title_fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set x-axis labels with rotation
            ax.set_xticks(x_positions)
            ax.set_xticklabels(education_occupation.index, rotation=45, ha='right')

        except Exception as e:
            print(f"Error in grouped bar chart: {e}")
            ax.set_title('Error: Could not create chart', color='red')


def main():
    """Main function to run the analysis"""
    try:
        analyzer = AdultDataAnalyzer('adult.csv')

        # Perform data cleaning
        analyzer.load_and_clean_data()

        # Create visualizations
        analyzer.create_visualizations()

    except Exception as e:
        print(f"Analysis failed: {e}")
        print("Please check that:")
        print("1. The 'adult.csv' file exists in the current directory")
        print("2. The file contains the required columns")
        print("3. The file is not corrupted")


if __name__ == "__main__":
    main()

