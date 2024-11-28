from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from finacial_models.model_manager import ModelManager


class UserInterface:
    def __init__(self):
        """Initialize the finance management system."""
        st.set_page_config(page_title="Personal Finance Management System", layout="wide")
        self.setup_session_state()
        self.model_manager = ModelManager()

    def setup_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'models' not in st.session_state:
            st.session_state.models = {}

    def main(self):
        """Main application interface and navigation."""
        st.title("Personal Finance Management System")

        page = st.sidebar.selectbox(
            "Select a Page",
            ["Data Input & Analysis", "Forecasting", "Financial Planning", "Recommendations"]
        )

        pages = {
            "Data Input & Analysis": self.data_input_page,
            "Forecasting": self.forecasting_page,
            "Financial Planning": self.financial_planning_page,
            "Recommendations": self.recommendations_page
        }
        pages[page]()

    def data_input_page(self):
        """Handle data input and initial analysis."""
        st.header("Data Input & Analysis")

        col1, col2 = st.columns(2)

        with col1:
            self._handle_data_input()

        with col2:
            if st.session_state.data is not None:
                self._display_data_visualization()

    def _handle_data_input(self):
        """Process user input for financial data."""
        st.subheader("Enter Your Financial Data")
        num_months = st.slider("Number of months of historical data:", 3, 24, 12)

        data_dict = {
            'Monthly Income (£)': [],
            'Total Expenses': [],
            'Savings for Property (£)': []
        }

        with st.expander("Enter Monthly Financial Data", expanded=True):
            for month in range(num_months):
                st.write(f"Month {month + 1}")

                # Income input
                income = st.number_input(
                    f"Income Month {month + 1}",
                    min_value=0.0,
                    value=5000.0,
                    key=f"income_{month}"
                )
                data_dict['Monthly Income (£)'].append(income)

                # Expenses input
                expenses = st.number_input(
                    f"Expenses Month {month + 1}",
                    min_value=0.0,
                    value=987.87,
                    key=f"expenses_{month}"
                )
                data_dict['Total Expenses'].append(expenses)

                # Savings input
                savings = st.number_input(
                    f"Property Savings Month {month + 1}",
                    min_value=0.0,
                    value=300.0,
                    key=f"savings_{month}"
                )
                data_dict['Savings for Property (£)'].append(savings)

        if st.button("Process Data"):
            df = pd.DataFrame(data_dict)
            df.index = pd.date_range(start='today', periods=num_months, freq='ME')
            st.session_state.data = df
            st.success("Data processed successfully!")

    def _display_data_visualization(self):
        """Create and display financial visualizations."""
        st.subheader("Data Visualization")

        fig = go.Figure()
        for column, name in [
            ('Monthly Income (£)', 'Income'),
            ('Total Expenses', 'Expenses'),
            ('Savings for Property (£)', 'Savings')
        ]:
            fig.add_trace(go.Scatter(
                x=st.session_state.data.index,
                y=st.session_state.data[column],
                mode='lines+markers',
                name=name
            ))

        fig.update_layout(
            title="Financial Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Amount (£)",
            hovermode='x'
        )
        st.plotly_chart(fig)

        if len(st.session_state.data) >= 3:
            self._perform_clustering_analysis()

    def _perform_clustering_analysis(self):
        """Perform and visualize clustering analysis."""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(st.session_state.data)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        st.subheader("Spending Pattern Clusters")
        fig = px.scatter_3d(
            st.session_state.data,
            x='Monthly Income (£)',
            y='Total Expenses',
            z='Savings for Property (£)',
            color=clusters,
            title="Financial Behavior Clusters"
        )
        st.plotly_chart(fig)

    def forecasting_page(self):
        """Handle financial forecasting functionality."""
        st.header("Financial Forecasting")

        if st.session_state.data is None:
            st.warning("Please input data first!")
            return

        if st.button("Train Models"):
            with st.spinner("Training models..."):
                self.model_manager.train_all_models(
                    st.session_state.data,
                    'Savings for Property (£)'
                )
                st.success("Models trained successfully!")

        forecast_period = st.slider("Forecast period (months):", 1, 12, 6)

        if self.model_manager.has_trained_models():
            future_dates = pd.date_range(
                start=st.session_state.data.index[-1],
                periods=forecast_period + 1,
                freq='ME'
            )[1:]

            self._display_forecasts(future_dates)

    def _display_forecasts(self, future_dates):
        """Display forecasting results."""
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=st.session_state.data.index,
            y=st.session_state.data['Savings for Property (£)'],
            mode='lines+markers',
            name='Historical'
        ))

        # Model predictions
        for model_name in self.model_manager.get_available_models():
            try:
                forecast = self.model_manager.get_prediction(model_name, len(future_dates))
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast['prediction'],
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(dash='dash')
                ))
            except Exception as e:
                st.error(f"Error generating forecast for {model_name}: {str(e)}")

        fig.update_layout(
            title="Savings Forecast",
            xaxis_title="Date",
            yaxis_title="Amount (£)",
            hovermode='x'
        )
        st.plotly_chart(fig)

    def financial_planning_page(self):
        """Handle financial planning and goal setting."""
        st.header("Financial Planning")

        if st.session_state.data is None:
            st.warning("Please input data first!")
            return

        st.subheader("Set Your Financial Goals")
        savings_goal = st.number_input("Savings Goal (£):", min_value=0.0, value=10000.0)
        target_date = st.date_input("Target Date:", datetime.now() + timedelta(days=365))

        days_to_goal = (target_date - datetime.now().date()).days
        months_to_goal = max(1, days_to_goal // 30)
        current_savings = st.session_state.data['Savings for Property (£)'].sum()
        required_monthly_savings = (savings_goal - current_savings) / months_to_goal

        col1, col2 = st.columns(2)
        with col1:
            st.write("Current Status:")
            st.write(f"Current Total Savings: £{current_savings:,.2f}")
            st.write(f"Required Monthly Savings: £{required_monthly_savings:,.2f}")

        with col2:
            progress = (current_savings / savings_goal) * 100
            st.write("Progress Tracking:")
            st.progress(min(progress / 100, 1.0))
            st.write(f"Progress: {progress:.1f}%")

    def recommendations_page(self):
        """Provide financial recommendations and analysis."""
        st.header("Financial Recommendations")

        if st.session_state.data is None:
            st.warning("Please input data first!")
            return

        income_mean = st.session_state.data['Monthly Income (£)'].mean()
        expenses_mean = st.session_state.data['Total Expenses'].mean()
        savings_mean = st.session_state.data['Savings for Property (£)'].mean()

        st.subheader("Financial Analysis")
        col1, col2 = st.columns(2)

        with col1:
            savings_rate = (savings_mean / income_mean) * 100
            st.write(f"Current Savings Rate: {savings_rate:.1f}%")
            if savings_rate < 20:
                st.warning("Consider increasing your savings rate to at least 20% of your income.")
            else:
                st.success("Good job! You're maintaining a healthy savings rate.")

        with col2:
            expense_ratio = (expenses_mean / income_mean) * 100
            st.write(f"Expense to Income Ratio: {expense_ratio:.1f}%")
            if expense_ratio > 50:
                st.warning("Your expenses are high relative to your income.")
                st.write("Consider:")
                st.write("- Review and cut non-essential expenses")
                st.write("- Look for ways to increase income")
                st.write("- Create a detailed budget")

        st.subheader("Scenario Planning")
        expense_reduction = st.slider(
            "Reduce monthly expenses by (%):",
            min_value=0,
            max_value=50,
            value=10
        )

        new_expenses = expenses_mean * (1 - expense_reduction / 100)
        additional_savings = expenses_mean - new_expenses

        st.write(f"If you reduce expenses by {expense_reduction}%:")
        st.write(f"- Monthly savings would increase by: £{additional_savings:,.2f}")
        st.write(f"- Annual additional savings: £{additional_savings * 12:,.2f}")


if __name__ == "__main__":
    app = UserInterface()
    app.main()