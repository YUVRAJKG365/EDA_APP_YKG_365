import streamlit as st
import pandas as pd
import duckdb
import io
import zipfile
import os
import tempfile
import re
import time
import plotly.express as px
import altair as alt
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from utils.session_state_manager import get_session_manager

# Get the session manager instance
session_manager = get_session_manager()

def sqlq_ui():
    st.markdown("## 🗃️ Advanced SQL Query Tool for CSV/Excel")
    
    # Section name for isolation
    section = "SQL Query"
    
    # --- File Upload Section ---
    st.markdown("### 📤 Upload & Auto-Convert Files to SQL Tables")
    
    # Use the session manager's file uploader key for this section
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Each file will be converted to a SQL table automatically",
        key=session_manager.get_file_uploader_key(section))
    
    # Initialize session state for tables using the session manager
    if not session_manager.has_data(section, 'sqlq_tables'):
        if session_manager.has_data(section, 'df') and session_manager.get_data(section, 'is_structured', False):
            session_manager.set_data(section, 'sqlq_tables', {"data": session_manager.get_dataframe(section).copy()})
        else:
            session_manager.set_data(section, 'sqlq_tables', {})

    # Show upload prompt if no files
    if (not uploaded_files and 
        (not session_manager.has_data(section, 'df') or 
         not session_manager.get_data(section, 'is_structured', False))):
        st.info("Please upload a CSV or Excel file to use SQL features.")
        st.info(
            "Run advanced SQL queries on your uploaded data with full visualization and analysis capabilities. "
            "Create tables with constraints, perform joins, generate insights, and visualize results - all in one tool."
        )
        st.info(
            """
            **How to use:**
            - Use the selectbox below to choose a table for preview and direct operations (insert/update/delete/select).
            - To create a new table, use a `CREATE TABLE` statement in the SQL box.
            - To insert into a table, select it in the box, then use `INSERT INTO tablename ...` in the SQL box.
            - For joins, views, or any advanced SQL, just write your query as usual—no need to select a table.
            - To create and insert into a new table in one go, you can use multiple statements separated by `;`.
            - All created tables will appear in the selectbox for further operations.
            """
        )
        st.info(
            "💡 **Tip:** If you want to create a table and then insert into it, run both statements together, separated by a semicolon (`;`).\n\n"
            "Example:\n"
            "`CREATE TABLE yu (name VARCHAR); INSERT INTO yu VALUES ('Yuvraj');`"
        )
        with st.expander("💡 SQL Syntax Help & Advanced Examples", expanded=False):
            st.markdown("""
    **Table creation with constraints:**
    ```sql
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR UNIQUE NOT NULL,
        email VARCHAR DEFAULT 'none@example.com',
        age INTEGER CHECK (age >= 0)
    );
    ```
    **Insert with auto-increment:**
    ```sql
    INSERT INTO users (username, email, age) VALUES ('alice', 'alice@example.com', 30);
    ```
    **Alter table:**
    ```sql
    ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
    ```
    **Drop table:**
    ```sql
    DROP TABLE IF EXISTS users;
    ```
    **Other features:**
    - `IDENTITY` is also supported for auto-increment: `id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY`
    - All standard SQL types: `INTEGER`, `BIGINT`, `FLOAT`, `DOUBLE`, `VARCHAR`, `DATE`, `TIMESTAMP`, etc.
    - Constraints: `PRIMARY KEY`, `UNIQUE`, `NOT NULL`, `DEFAULT`, `CHECK`
    - All DML/DDL: `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, `RENAME`, etc.
    - Multiple statements: Separate with `;`
                """)
        st.info("DuckDB supports most standard SQL features. Table name `data` is your uploaded file.")
        
        # Stop execution here if no files are uploaded
        st.stop()
    
    # Process uploaded files if any
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Sanitize table name from filename
                table_name = re.sub(r'[^a-zA-Z0-9_]', '_', uploaded_file.name.split('.')[0])
                if table_name[0].isdigit():
                    table_name = f"t_{table_name}"  # Add prefix if starts with digit

                # Read file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # Excel
                    df = pd.read_excel(uploaded_file)

                # Store in session state using session manager
                sqlq_tables = session_manager.get_data(section, 'sqlq_tables', {})
                sqlq_tables[table_name] = df
                session_manager.set_data(section, 'sqlq_tables', sqlq_tables)
                st.success(f"✅ Table '{table_name}' created from {uploaded_file.name} with {df.shape[0]} rows")

            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")

    # Check if we have any tables
    sqlq_tables = session_manager.get_data(section, 'sqlq_tables', {})
    if not sqlq_tables:
        st.info("Please upload files to use SQL features")
        return

    # --- Table Selection ---
    table_names = list(sqlq_tables.keys())
    selected_table = st.selectbox(
        "Select a table to preview:",
        table_names,
        key=f"{section}_table_select"
    )
    df = sqlq_tables[selected_table]

    # --- Main Navigation Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Table Info", 
        "🔧 No-Code Tools", 
        "📝 SQL Editor", 
        "📊 Visualizations", 
        "✂️ Create Dataset", 
        "⚙️ Management"
    ])

    # --- Tab 1: Table Information ---
    with tab1:
        st.markdown(f"### 🧾 Table Structure: {selected_table}")

        # --- Schema information ---
        with st.expander("#### Schema Information"):
            schema_df = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(df[col].dtype) for col in df.columns],
                "Nulls": df.isnull().sum().values,
                "Unique": [df[col].nunique() for col in df.columns]
            })
            st.dataframe(schema_df, hide_index=True)
            st.markdown(f"**Rows:** {df.shape[0]:,} &nbsp;&nbsp; **Columns:** {df.shape[1]}")

        # --- Sample data ---
        with st.expander("#### Sample Data"):
            st.dataframe(df.head(10), height=350)
            if st.button("Show Full Data", key=f"show_full_{selected_table}"):
                st.dataframe(df, height=500)

        # --- Dataset Summary (No Expander, No Columns) ---
        with st.expander("### Dataset Summary"):
            # Column Types
            st.markdown("**Column Types**")
            type_counts = pd.DataFrame({
                "Type": [str(dtype) for dtype in df.dtypes],
                "Count": 1
            }).groupby("Type").count().reset_index()
            st.dataframe(type_counts, hide_index=True)

            # Missing Values
            st.markdown("**Missing Values**")
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Column", "Missing"]
            missing["Pct"] = (missing["Missing"] / len(df)) * 100
            st.dataframe(missing, hide_index=True)

            # Numeric Columns Summary
            with st.container():
                st.markdown("**Numeric Columns Summary**")
                numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
                if numeric_cols:
                    num_summary = df[numeric_cols].describe().T
                    st.dataframe(num_summary, use_container_width=True)
                else:
                    st.info("No numeric columns found")

            # Categorical Columns Summary
            with st.container():
                st.markdown("**Categorical Columns Summary**")
                cat_cols = [col for col in df.columns if col not in numeric_cols]
                if cat_cols:
                    cat_summary = pd.DataFrame({
                        "Column": cat_cols,
                        "Unique": [df[col].nunique() for col in cat_cols],
                        "Top": [df[col].value_counts().index[0] for col in cat_cols],
                        "Freq": [df[col].value_counts().iloc[0] for col in cat_cols]
                    })
                    st.dataframe(cat_summary, hide_index=True)
                else:
                    st.info("No categorical columns found")

    # --- Tab 2: No-Code Tools ---
    with tab2:
        st.markdown("### 🔧 No-Code SQL Query Builder")
        st.info("Build queries visually without writing SQL")

        try:
            # Initialize filter_conditions in session state if not exists
            if not session_manager.has_data(section, 'filter_conditions'):
                session_manager.set_data(section, 'filter_conditions', [])

            # --- Table Selection Section ---
            st.markdown("#### 1️⃣ Select Tables")
            join_type = st.selectbox(
                "Query Type",
                ["Single Table", "Join Tables"],
                key=f"{section}_query_type"
            )

            # Get available tables with error handling
            available_tables = list(sqlq_tables.keys())
            if not available_tables:
                st.warning("No tables available. Please upload data first.")
                st.stop()

            if join_type == "Single Table":
                # Single table selection
                builder_table = st.selectbox("Select table:", available_tables, key=f"{section}_builder_table")
                try:
                    builder_df = sqlq_tables[builder_table]
                    from_clause = builder_table
                    selected_table = builder_table  # Store for prebuilt queries
                except KeyError:
                    st.error(f"Table '{builder_table}' not found in session state")
                    st.stop()
            else:
                # Multi-table join configuration
                join_cols = st.columns([1, 1, 2])

                with join_cols[0]:
                    left_table = st.selectbox("Left Table", available_tables, key=f"{section}_left_table_join")
                    try:
                        left_df = sqlq_tables[left_table]
                        left_columns = left_df.columns.tolist()
                    except KeyError:
                        st.error(f"Table '{left_table}' not found in session state")
                        st.stop()

                with join_cols[1]:
                    join_method = st.selectbox(
                        "Join Type",
                        ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"],
                        key=f"{section}_join_method"
                    )
                    right_table_options = [t for t in available_tables if t != left_table]
                    if not right_table_options:
                        st.error("No other tables available for joining")
                        st.stop()

                    right_table = st.selectbox("Right Table", right_table_options, key=f"{section}_right_table_join")
                    try:
                        right_df = sqlq_tables[right_table]
                        right_columns = right_df.columns.tolist()
                    except KeyError:
                        st.error(f"Table '{right_table}' not found in session state")
                        st.stop()

                with join_cols[2]:
                    st.markdown("**Join Conditions**")
                    join_condition_cols = st.columns(2)
                    with join_condition_cols[0]:
                        if not left_columns:
                            st.error("No columns found in left table")
                            st.stop()
                        left_join_col = st.selectbox("Left Column", left_columns, key=f"{section}_left_join_col")
                    with join_condition_cols[1]:
                        if not right_columns:
                            st.error("No columns found in right table")
                            st.stop()
                        right_join_col = st.selectbox("Right Column", right_columns, key=f"{section}_right_join_col")

                    additional_conditions = st.text_input("Additional Join Conditions (e.g., a.col1 > b.col2)", "", 
                                                         key=f"{section}_additional_conditions")

                from_clause = f"{left_table} a {join_method} {right_table} b ON a.{left_join_col} = b.{right_join_col}"
                if additional_conditions:
                    from_clause += f" AND {additional_conditions}"

                try:
                    builder_df = pd.concat([left_df, right_df], axis=1)
                    selected_table = left_table  # Store for prebuilt queries
                except Exception as e:
                    st.error(f"Failed to combine tables: {str(e)}")
                    st.stop()

            # --- Column Selection ---
            st.markdown("#### 2️⃣ Select Columns")
            if builder_df.empty:
                st.warning("Selected table is empty")
                st.stop()

            available_columns = builder_df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to include:",
                available_columns,
                default=available_columns[:min(3, len(available_columns))],
                key=f"{section}_selected_columns"
            )

            # --- Filter Conditions ---
            st.markdown("#### 3️⃣ Add Filters (WHERE conditions)")
            filter_cols = st.columns(3)
            filter_conditions = session_manager.get_data(section, 'filter_conditions', [])

            with filter_cols[0]:
                filter_column = st.selectbox("Column", available_columns, key=f"{section}_filter_col")

            with filter_cols[1]:
                filter_operator = st.selectbox("Operator",
                                               ["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "IS NULL", "IS NOT NULL",
                                                "BETWEEN"],
                                               key=f"{section}_filter_op")
            with filter_cols[2]:
                if filter_operator not in ["IS NULL", "IS NOT NULL"]:
                    if filter_operator == "BETWEEN":
                        try:
                            if is_numeric_dtype(builder_df[filter_column]):
                                filter_value1 = st.number_input("Value 1", key=f"{section}_filter_val1")
                                filter_value2 = st.number_input("Value 2", key=f"{section}_filter_val2")
                            else:
                                filter_value1 = st.text_input("Value 1", key=f"{section}_filter_val1")
                                filter_value2 = st.text_input("Value 2", key=f"{section}_filter_val2")
                        except KeyError:
                            st.error(f"Column '{filter_column}' not found in dataframe")
                            st.stop()
                    else:
                        try:
                            if is_numeric_dtype(builder_df[filter_column]):
                                filter_value = st.number_input("Value", key=f"{section}_filter_val")
                            else:
                                filter_value = st.text_input("Value", key=f"{section}_filter_val")
                        except KeyError:
                            st.error(f"Column '{filter_column}' not found in dataframe")
                            st.stop()

            if st.button("Add Filter", key=f"{section}_add_filter"):
                try:
                    if filter_operator in ["IS NULL", "IS NOT NULL"]:
                        filter_conditions.append(f"{filter_column} {filter_operator}")
                    elif filter_operator == "BETWEEN":
                        val1 = f"'{filter_value1}'" if isinstance(filter_value1, str) else str(filter_value1)
                        val2 = f"'{filter_value2}'" if isinstance(filter_value2, str) else str(filter_value2)
                        filter_conditions.append(f"{filter_column} BETWEEN {val1} AND {val2}")
                    else:
                        if isinstance(filter_value, str) and filter_operator not in ["IN"]:
                            filter_value = f"'{filter_value}'"
                        filter_conditions.append(f"{filter_column} {filter_operator} {filter_value}")

                    session_manager.set_data(section, 'filter_conditions', filter_conditions)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add filter: {str(e)}")

            # Show current filters
            if filter_conditions:
                st.markdown("**Current Filters:**")
                for i, condition in enumerate(filter_conditions):
                    cols = st.columns([0.9, 0.1])
                    cols[0].code(condition, language="sql")
                    if cols[1].button("❌", key=f"{section}_remove_filter_{i}"):
                        try:
                            filter_conditions.pop(i)
                            session_manager.set_data(section, 'filter_conditions', filter_conditions)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove filter: {str(e)}")

            # --- Aggregations ---
            st.markdown("#### 4️⃣ Aggregations (Optional)")
            agg_cols = st.columns(2)
            with agg_cols[0]:
                agg_column = st.selectbox("Column to aggregate", available_columns, key=f"{section}_agg_col")
            with agg_cols[1]:
                agg_func = st.selectbox("Function", ["", "COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE"],
                                        key=f"{section}_agg_func")

            # --- Group By ---
            st.markdown("#### 5️⃣ Group By (Optional)")
            group_by = st.multiselect("GROUP BY columns:", available_columns, key=f"{section}_group_by")

            # --- Order By ---
            st.markdown("#### 6️⃣ Order By (Optional)")
            order_cols = st.columns(2)
            with order_cols[0]:
                order_by = st.selectbox("Column to sort by:", [""] + available_columns, key=f"{section}_order_by")
            with order_cols[1]:
                order_dir = st.radio("Direction:", ["ASC", "DESC"], horizontal=True, key=f"{section}_order_dir")

            # --- Generate SQL ---
            st.markdown("#### 7️⃣ Generated SQL Query")
            if selected_columns:
                try:
                    select_clause = ", ".join(selected_columns)
                    if agg_func and agg_column:
                        select_clause = select_clause.replace(agg_column,
                                                              f"{agg_func}({agg_column}) AS {agg_func.lower()}_{agg_column}")

                    where_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ""
                    group_clause = f"GROUP BY {', '.join(group_by)}" if group_by else ""
                    order_clause = f"ORDER BY {order_by} {order_dir}" if order_by else ""

                    generated_sql = f"SELECT {select_clause} FROM {from_clause} {where_clause} {group_clause} {order_clause} LIMIT 1000"

                    st.code(generated_sql, language="sql")

                    if st.button("▶️ Run Generated Query", type="primary", key=f"{section}_run_generated"):
                        try:
                            # Store the query in session state
                            session_manager.set_data(section, 'sql_query', generated_sql)

                            # Start with the appropriate dataframe
                            if join_type == "Single Table":
                                query_df = builder_df.copy()
                            else:
                                query_df = builder_df.copy()

                            # Apply WHERE conditions
                            if filter_conditions:
                                try:
                                    for condition in filter_conditions:
                                        # Convert SQL-like condition to pandas query format
                                        pandas_condition = condition
                                        pandas_condition = pandas_condition.replace("=", "==")
                                        pandas_condition = pandas_condition.replace("!=", "!=")
                                        pandas_condition = pandas_condition.replace("IS NULL", "isnull()")
                                        pandas_condition = pandas_condition.replace("IS NOT NULL", "notnull()")
                                        query_df = query_df.query(pandas_condition)
                                except Exception as e:
                                    st.error(f"Error applying filter condition '{condition}': {str(e)}")
                                    st.stop()

                            # Apply GROUP BY and aggregations
                            if group_by and agg_func and agg_column:
                                try:
                                    agg_dict = {agg_column: agg_func.lower()}
                                    query_df = query_df.groupby(group_by).agg(agg_dict).reset_index()
                                    # Rename the aggregated column
                                    query_df = query_df.rename(columns={agg_column: f"{agg_func.lower()}_{agg_column}"})
                                except Exception as e:
                                    st.error(f"Error applying aggregation: {str(e)}")
                                    st.stop()

                            # Apply ORDER BY
                            if order_by:
                                try:
                                    ascending = order_dir == "ASC"
                                    query_df = query_df.sort_values(by=order_by, ascending=ascending)
                                except Exception as e:
                                    st.error(f"Error applying sorting: {str(e)}")
                                    st.stop()

                            # Select only the requested columns
                            try:
                                query_df = query_df[selected_columns]
                            except KeyError as e:
                                st.error(f"Column not found in results: {str(e)}")
                                st.stop()

                            # Apply LIMIT
                            query_df = query_df.head(1000)

                            # Display results
                            st.success("✅ Query executed successfully!")
                            st.write("### Query Results")
                            st.dataframe(query_df)

                            # Show some stats
                            st.write(f"**Results:** {len(query_df)} rows returned")

                            # Store the result in session state
                            session_manager.set_data(section, 'last_query_result', query_df)
                            session_manager.set_data(section, 'last_query', generated_sql)

                            # Clear filters after running
                            session_manager.set_data(section, 'filter_conditions', [])

                        except Exception as e:
                            st.error(f"Failed to execute query: {str(e)}")
                            st.stop()

                except Exception as e:
                    st.error(f"Failed to generate SQL: {str(e)}")

        except Exception as e:
            st.error(f"An unexpected error occurred in the query builder: {str(e)}")
            st.stop()

        # --- Prebuilt Analytical Queries ---
        try:
            st.markdown("---")
            st.markdown("### 📊 Prebuilt Analytical Queries")
            st.markdown("Run common analytical queries with one click")

            # Initialize session state variables if they don't exist
            if not session_manager.has_data(section, 'prebuilt_expanded'):
                session_manager.set_data(section, 'prebuilt_expanded', {
                    'basic_analysis': True,
                    'advanced_analysis': True
                })
            if not session_manager.has_data(section, 'prebuilt_selections'):
                session_manager.set_data(section, 'prebuilt_selections', {
                    'top_col': None,
                    'date_col': None,
                    'value_col': None
                })

            # Ensure we have tables available
            if not sqlq_tables:
                st.warning("No tables available. Please upload data first.")
                st.stop()

            selected_table = list(sqlq_tables.keys())[0]  # Default to first table

            # Basic Analysis expander
            prebuilt_expanded = session_manager.get_data(section, 'prebuilt_expanded')
            with st.expander("**Basic Analysis**",
                             expanded=prebuilt_expanded.get('basic_analysis', True)):
                if st.button("Show Table Info"):
                    try:
                        if selected_table not in sqlq_tables:
                            raise ValueError(f"Table '{selected_table}' not found in session state")

                        df = sqlq_tables[selected_table]
                        if df.empty:
                            st.warning("Selected table is empty")
                            st.stop()

                        query = f"""
                            SELECT 
                                COUNT(*) AS row_count,
                                {', '.join([f'COUNT(DISTINCT {col}) AS {col}_unique' for col in df.columns])}
                            FROM {selected_table}
                            """
                        session_manager.set_data(section, 'sql_query', query)
                        result_df = pd.DataFrame({
                            'Metric': ['Row Count'] + [f'{col} (Unique)' for col in df.columns],
                            'Value': [len(df)] + [df[col].nunique() for col in df.columns]
                        })
                        st.write("### Table Information")
                        st.dataframe(result_df)
                    except Exception as e:
                        st.error(f"Failed to generate table info: {str(e) if str(e) else 'Unknown error'}")

                if st.button("Count Nulls per Column"):
                    try:
                        if selected_table not in sqlq_tables:
                            raise ValueError(f"Table '{selected_table}' not found in session state")

                        df = sqlq_tables[selected_table]
                        if df.empty:
                            st.warning("Selected table is empty")
                            st.stop()

                        query = "SELECT " + ", ".join(
                            [f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}_nulls"
                             for col in df.columns]) + f" FROM {selected_table}"
                        session_manager.set_data(section, 'sql_query', query)
                        result_df = pd.DataFrame({
                            'Column': df.columns,
                            'Null Count': [df[col].isnull().sum() for col in df.columns]
                        })
                        st.write("### Null Values Count")
                        st.dataframe(result_df)
                    except Exception as e:
                        st.error(f"Failed to count nulls: {str(e) if str(e) else 'Unknown error'}")

                if st.button("Numeric Summary"):
                    try:
                        if selected_table not in sqlq_tables:
                            raise ValueError(f"Table '{selected_table}' not found in session state")

                        df = sqlq_tables[selected_table]
                        if df.empty:
                            st.warning("Selected table is empty")
                            st.stop()

                        numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
                        if not numeric_cols:
                            st.info("No numeric columns in this table")
                            st.stop()

                        query = "SELECT " + ", ".join(
                            [f"MIN({col}) AS {col}_min, MAX({col}) AS {col}_max, AVG({col}) AS {col}_avg"
                             for col in numeric_cols]) + f" FROM {selected_table}"
                        session_manager.set_data(section, 'sql_query', query)
                        result_data = []
                        for col in numeric_cols:
                            result_data.append({
                                'Column': col,
                                'Min': df[col].min(),
                                'Max': df[col].max(),
                                'Average': df[col].mean(),
                                'Std Dev': df[col].std()
                            })
                        result_df = pd.DataFrame(result_data)
                        st.write("### Numeric Columns Summary")
                        st.dataframe(result_df)
                    except Exception as e:
                        st.error(f"Failed to generate numeric summary: {str(e) if str(e) else 'Unknown error'}")

            # Advanced Analysis expander
            with st.expander("**Advanced Analysis**",
                             expanded=prebuilt_expanded.get('advanced_analysis', True)):
                try:
                    if selected_table not in sqlq_tables:
                        raise ValueError(f"Table '{selected_table}' not found in session state")

                    df = sqlq_tables[selected_table]
                    if df.empty:
                        st.warning("Selected table is empty")
                        st.stop()

                    # Top 10 Values by Column
                    prebuilt_selections = session_manager.get_data(section, 'prebuilt_selections')
                    current_top_col = prebuilt_selections.get('top_col',
                                                           df.columns[0] if not df.empty else None)
                    top_col = st.selectbox(
                        "Select column:",
                        df.columns,
                        key=f"{section}_top_col_select",
                        index=df.columns.get_loc(current_top_col) if current_top_col in df.columns else 0
                    )

                    if st.button("Top 10 Values by Column"):
                        try:
                            prebuilt_selections['top_col'] = top_col
                            session_manager.set_data(section, 'prebuilt_selections', prebuilt_selections)
                            query = f"""
                                SELECT {top_col}, COUNT(*) AS count 
                                FROM {selected_table} 
                                GROUP BY {top_col} 
                                ORDER BY count DESC 
                                LIMIT 10
                                """
                            session_manager.set_data(section, 'sql_query', query)
                            result_df = df[top_col].value_counts().head(10).reset_index()
                            result_df.columns = [top_col, 'Count']
                            st.write(f"### Top 10 Values in {top_col}")
                            st.dataframe(result_df)
                        except Exception as e:
                            st.error(f"Failed to generate top values: {str(e) if str(e) else 'Unknown error'}")

                    # Time Series Analysis
                    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                    if date_cols:
                        current_date_col = prebuilt_selections.get('date_col',
                                                                date_cols[0] if date_cols else None)
                        date_col = st.selectbox(
                            "Select date column:",
                            date_cols,
                            key=f"{section}_date_col_select",
                            index=date_cols.index(current_date_col) if current_date_col in date_cols else 0
                        )

                        value_col_options = [col for col in df.columns if col != date_col]
                        current_value_col = prebuilt_selections.get('value_col', value_col_options[
                            0] if value_col_options else None)
                        value_col = st.selectbox(
                            "Select value column:",
                            value_col_options,
                            key=f"{section}_value_col_select",
                            index=value_col_options.index(
                                current_value_col) if current_value_col in value_col_options else 0
                        )

                        if st.button("Time Series Analysis"):
                            try:
                                prebuilt_selections['date_col'] = date_col
                                prebuilt_selections['value_col'] = value_col
                                session_manager.set_data(section, 'prebuilt_selections', prebuilt_selections)
                                query = f"""
                                    SELECT 
                                        DATE({date_col}) AS day, 
                                        SUM({value_col}) AS total_{value_col},
                                        AVG({value_col}) AS avg_{value_col}
                                    FROM {selected_table}
                                    GROUP BY DATE({date_col})
                                    ORDER BY day
                                    """
                                session_manager.set_data(section, 'sql_query', query)
                                df[date_col] = pd.to_datetime(df[date_col])
                                result_df = df.groupby(df[date_col].dt.date)[value_col].agg(
                                    ['sum', 'mean']).reset_index()
                                result_df.columns = ['Date', f'Total {value_col}', f'Average {value_col}']
                                st.write(f"### Time Series Analysis of {value_col} by {date_col}")
                                st.dataframe(result_df)
                            except Exception as e:
                                st.error(f"Failed to generate time series: {str(e) if str(e) else 'Unknown error'}")
                    else:
                        st.info("No date columns in this table")

                    # Correlation Matrix
                    if st.button("Correlation Matrix"):
                        try:
                            numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
                            if len(numeric_cols) >= 2:
                                query = f"""
                                    SELECT {', '.join(numeric_cols)}
                                    FROM {selected_table}
                                    """
                                session_manager.set_data(section, 'sql_query', query)
                                corr_matrix = df[numeric_cols].corr()
                                st.write("### Correlation Matrix")
                                st.dataframe(
                                    corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
                            else:
                                st.info("Need at least 2 numeric columns for correlation")
                        except Exception as e:
                            st.error(f"Failed to generate correlation: {str(e) if str(e) else 'Unknown error'}")

                except Exception as e:
                    st.error(f"Error in advanced analysis section: {str(e) if str(e) else 'Unknown error'}")

        except Exception as e:
            error_msg = str(e) if str(e) else "An unknown error occurred"
            st.error(f"An unexpected error occurred in prebuilt queries: {error_msg}")
            st.stop()

    # --- Tab 3: SQL Editor ---
    with tab3:
        st.markdown("### 📝 SQL Query Editor")
        with st.expander("💡 SQL Syntax Help & Advanced Examples", expanded=False):
            st.markdown("""
    **Table creation with constraints:**
    ```sql
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR UNIQUE NOT NULL,
        email VARCHAR DEFAULT 'none@example.com',
        age INTEGER CHECK (age >= 0)
    );
    ```
    **Insert with auto-increment:**
    ```sql
    INSERT INTO users (username, email, age) VALUES ('alice', 'alice@example.com', 30);
    ```
    **Alter table:**
    ```sql
    ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
    ```
    **Drop table:**
    ```sql
    DROP TABLE IF EXISTS users;
    ```
    **Other features:**
    - `IDENTITY` is also supported for auto-increment: `id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY`
    - All standard SQL types: `INTEGER`, `BIGINT`, `FLOAT`, `DOUBLE`, `VARCHAR`, `DATE`, `TIMESTAMP`, etc.
    - Constraints: `PRIMARY KEY`, `UNIQUE`, `NOT NULL`, `DEFAULT`, `CHECK`
    - All DML/DDL: `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, `RENAME`, etc.
    - Multiple statements: Separate with `;`
            """)
            st.info("DuckDB supports most standard SQL features. Table name `data` is your uploaded file.")
            
        query = st.text_area(
            "Write your SQL query below:",
            value=session_manager.get_data(section, 'sql_query', f"SELECT * FROM {selected_table} LIMIT 10"),
            height=150,
            key=f"{section}_sql_query_input"
        )

        # Query execution
        if st.button("▶️ Run Query", type="primary", key=f"{section}_run_query_main"):
            try:
                start_time = time.time()

                # Connect to DuckDB and register all tables
                con = duckdb.connect(database=':memory:')
                for tname, tdf in sqlq_tables.items():
                    con.register(tname, tdf)

                # Execute query
                result = con.execute(query).fetchdf()
                exec_time = time.time() - start_time

                st.success(f"✅ Query executed successfully in {exec_time:.2f}s")
                st.dataframe(result, use_container_width=True, height=400)
                st.caption(
                    f"Rows: {result.shape[0]:,} | Columns: {result.shape[1]} | Memory: {result.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

                # Store result for visualization and export
                session_manager.set_data(section, 'last_query_result', result)
                session_manager.set_data(section, 'last_query', query)
                session_manager.set_data(section, 'last_query_time', exec_time)

                # Add to history
                query_history = session_manager.get_data(section, 'query_history', [])
                query_history.append({
                    "query": query,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "rows": result.shape[0],
                    "cols": result.shape[1]
                })
                session_manager.set_data(section, 'query_history', query_history)

                con.close()

            except Exception as e:
                st.info(f"❌ Query failed: {str(e)}")

        # --- Option to update a table with the result ---
        last_query_result = session_manager.get_data(section, 'last_query_result')
        if last_query_result is not None:
            update_table_name = st.text_input("Update or create table with result (enter table name):", 
                                             value=selected_table, 
                                             key=f"{section}_update_table_name")
            if st.button("💾 Save Result to Table", key=f"{section}_update_df_btn"):
                sqlq_tables[update_table_name] = last_query_result.copy()
                session_manager.set_data(section, 'sqlq_tables', sqlq_tables)
                st.success(f"Table '{update_table_name}' updated with SQL query result!")

        # --- Query History ---
        with st.expander("🕑 Query History", expanded=False):
            query_history = session_manager.get_data(section, 'query_history', [])
            if query_history:
                st.markdown("### Recent Queries")
                history_df = pd.DataFrame(query_history)
                st.dataframe(
                    history_df.sort_values("time", ascending=False).head(10),
                    column_config={
                        "time": "Time",
                        "query": st.column_config.TextColumn("Query", width="large"),
                        "rows": "Rows",
                        "cols": "Cols"
                    },
                    hide_index=True,
                    use_container_width=True
                )

                selected_history = st.selectbox(
                    "Select query to rerun:",
                    range(len(query_history)),
                    format_func=lambda x: query_history[x]["query"][:50] + "...",
                    index=0
                )

                if st.button("Rerun Selected Query"):
                    session_manager.set_data(section, 'sql_query', query_history[selected_history]["query"])
                    st.rerun()
            else:
                st.info("No query history yet")

    # --- Tab 4: Visualizations ---
    with tab4:
        last_query_result = session_manager.get_data(section, 'last_query_result')
        if last_query_result is not None:
            st.markdown("### 📊 Visualize Query Results")

            # --- Table selection for visualization ---
            viz_table_names = list(sqlq_tables.keys())
            selected_viz_table = st.selectbox(
                "Select table to visualize:",
                viz_table_names,
                key=f"{section}_viz_table_select"
            )
            result_df = sqlq_tables[selected_viz_table]

            # Auto-detect visualization types based on data
            numeric_cols = [col for col in result_df.columns if is_numeric_dtype(result_df[col])]
            date_cols = [col for col in result_df.columns if pd.api.types.is_datetime64_any_dtype(result_df[col])]
            cat_cols = [col for col in result_df.columns if col not in numeric_cols + date_cols]

            chart_type = st.selectbox("Chart Type",
                                      ["Bar", "Line", "Scatter", "Histogram", "Pie", "Heatmap"],
                                      index=0,
                                      key=f"{section}_chart_type")

            cols = st.columns(2)

            with cols[0]:
                if chart_type in ["Bar", "Line", "Scatter", "Pie"]:
                    x_axis = st.selectbox("X-axis", list(result_df.columns), 
                                         key=f"{section}_x_axis")
                    y_axis = st.selectbox("Y-axis", numeric_cols if numeric_cols else list(result_df.columns), 
                                         key=f"{section}_y_axis")
                    if chart_type == "Pie":
                        color_col = st.selectbox("Color by", [None] + list(result_df.columns),
                                               key=f"{section}_pie_color")
                    else:
                        color_col = st.selectbox("Color by", [None] + list(result_df.columns),
                                               key=f"{section}_color_col")
                elif chart_type == "Histogram":
                    col = st.selectbox("Column", numeric_cols if numeric_cols else result_df.columns,
                                     key=f"{section}_hist_col")
                    bins = st.slider("Bins", 5, 100, 20, key=f"{section}_bins")
                elif chart_type == "Heatmap":
                    x_axis = st.selectbox("X-axis", result_df.columns, 
                                         key=f"{section}_heatmap_x")
                    y_axis = st.selectbox("Y-axis", result_df.columns, 
                                         key=f"{section}_heatmap_y")
                    z_axis = st.selectbox("Value", numeric_cols if numeric_cols else result_df.columns, 
                                         key=f"{section}_heatmap_z")

            with cols[1]:
                if chart_type == "Bar":
                    orientation = st.radio("Orientation", ["Vertical", "Horizontal"],
                                         key=f"{section}_bar_orientation")
                    barmode = st.radio("Bar Mode", ["group", "stack"],
                                     key=f"{section}_bar_mode")
                elif chart_type == "Line":
                    line_mode = st.radio("Line Mode", ["lines", "lines+markers"],
                                       key=f"{section}_line_mode")
                elif chart_type == "Scatter":
                    size_col = st.selectbox("Size", [None] + numeric_cols,
                                          key=f"{section}_scatter_size")

            # Generate the chart
            if st.button("Generate Chart", key=f"{section}_generate_chart"):
                try:
                    if chart_type == "Bar":
                        if orientation == "Vertical":
                            fig = px.bar(result_df, x=x_axis, y=y_axis, color=color_col, barmode=barmode)
                        else:
                            fig = px.bar(result_df, y=x_axis, x=y_axis, color=color_col, barmode=barmode,
                                         orientation='h')
                    elif chart_type == "Line":
                        fig = px.line(result_df, x=x_axis, y=y_axis, color=color_col,
                                      markers=(line_mode == "lines+markers"))
                    elif chart_type == "Scatter":
                        fig = px.scatter(result_df, x=x_axis, y=y_axis, color=color_col, size=size_col)
                    elif chart_type == "Histogram":
                        fig = px.histogram(result_df, x=col, nbins=bins)
                    elif chart_type == "Pie":
                        fig = px.pie(result_df, names=x_axis, values=y_axis, color=color_col)
                    elif chart_type == "Heatmap":
                        pivot_df = result_df.groupby([x_axis, y_axis])[z_axis].mean().unstack()
                        fig = px.imshow(pivot_df, labels=dict(x=x_axis, y=y_axis, color=z_axis))

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Failed to generate chart: {str(e)}")

            # --- Export Results ---
            st.markdown("### 📥 Export Results")
            result_df = last_query_result

            export_cols = st.columns(3)

            with export_cols[0]:
                if st.button("Export to CSV", key=f"{section}_export_csv"):
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="query_result.csv",
                        mime="text/csv",
                        key=f"{section}_download_csv"
                    )

            with export_cols[1]:
                if st.button("Export to Excel", key=f"{section}_export_excel"):
                    excel_buffer = io.BytesIO()
                    result_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=excel_buffer,
                        file_name="query_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"{section}_download_excel"
                    )

            with export_cols[2]:
                if st.button("Export to JSON", key=f"{section}_export_json"):
                    json_str = result_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="query_result.json",
                        mime="application/json",
                        key=f"{section}_download_json"
                    )
        else:
            st.info("Run a query first to see visualization options")

    # --- Tab 5: Create Dataset ---
    with tab5:
        st.markdown("### ✂️ Create New Dataset from Selected Columns/Rows")
        orig_df = session_manager.get_dataframe(section) if session_manager.has_data(section, 'df') else df
        if orig_df is not None:
            columns = st.multiselect("Select columns", orig_df.columns.tolist(), 
                                   default=list(orig_df.columns),
                                   key=f"{section}_subset_cols")
            max_rows = len(orig_df)
            num_rows = st.number_input("Number of rows", min_value=1, max_value=max_rows, 
                                     value=min(10, max_rows), step=1,
                                     key=f"{section}_subset_rows")
            new_table_name = st.text_input("Name for new dataset/table", value="my_subset",
                                         key=f"{section}_new_table_name")
            if st.button("Create new dataset/table from selection", key=f"{section}_create_subset_btn"):
                subset_df = orig_df.loc[:num_rows-1, columns]
                sqlq_tables[new_table_name] = subset_df.copy()
                session_manager.set_data(section, 'sqlq_tables', sqlq_tables)
                st.success(f"Table '{new_table_name}' created with {len(subset_df)} rows and {len(columns)} columns.")
                st.dataframe(subset_df)

                # Download buttons (only show after subset is created)
                csv = subset_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"{new_table_name}.csv",
                    mime="text/csv",
                    key=f"{section}_download_subset_csv_{new_table_name}"
                )
                excel_buffer = io.BytesIO()
                subset_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                excel_buffer.seek(0)
                st.download_button(
                    label="Download as Excel",
                    data=excel_buffer,
                    file_name=f"{new_table_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"{section}_download_subset_excel_{new_table_name}"
                )

    # --- Tab 6: Management ---
    with tab6:
        st.markdown("### 🗄️ Table Management")

        # --- Multi-Table Operations ---
        if len(sqlq_tables) > 1:
            with st.expander("🔗 Multi-Table Operations", expanded=False):
                st.markdown("### Join Tables")

                join_cols = st.columns(3)

                with join_cols[0]:
                    left_table = st.selectbox("Left Table", table_names, key=f"{section}_left_table_mgmt")
                    left_df = sqlq_tables[left_table]
                    left_col = st.selectbox("Left Column", left_df.columns, key=f"{section}_left_col_mgmt")

                with join_cols[1]:
                    join_type = st.selectbox("Join Type", ["INNER", "LEFT", "RIGHT", "FULL"], 
                                           key=f"{section}_join_type_mgmt")
                    right_table = st.selectbox("Right Table", [t for t in table_names if t != left_table],
                                             key=f"{section}_right_table_mgmt")
                    right_df = sqlq_tables[right_table]
                    right_col = st.selectbox("Right Column", right_df.columns, key=f"{section}_right_col_mgmt")

                with join_cols[2]:
                    st.markdown("**Join Preview**")
                    join_sql = f"""
                    SELECT * FROM {left_table} a
                    {join_type} JOIN {right_table} b
                    ON a.{left_col} = b.{right_col}
                    LIMIT 10
                    """
                    st.code(join_sql, language="sql")

                if st.button("Preview Join", key=f"{section}_preview_join"):
                    try:
                        con = duckdb.connect(database=':memory:')
                        for tname, tdf in sqlq_tables.items():
                            con.register(tname, tdf)

                        result = con.execute(join_sql).fetchdf()
                        st.dataframe(result, height=300)

                        st.markdown("**Save Joined Table As:**")
                        new_name = st.text_input("New table name", "joined_table",
                                                key=f"{section}_joined_table_name")
                        if st.button("Save Joined Table", key=f"{section}_save_joined_table"):
                            sqlq_tables[new_name] = result
                            session_manager.set_data(section, 'sqlq_tables', sqlq_tables)
                            st.success(f"Table '{new_name}' saved!")

                        con.close()
                    except Exception as e:
                        st.info(f"Join failed: {str(e)}")

        # Create new table from query result
        if last_query_result is not None:
            new_table_name = st.text_input("Save result as new table:", "query_result",
                                         key=f"{section}_save_result_table")
            if st.button("💾 Save as New Table", key=f"{section}_save_result_btn"):
                sqlq_tables[new_table_name] = last_query_result.copy()
                session_manager.set_data(section, 'sqlq_tables', sqlq_tables)
                st.success(f"Table '{new_table_name}' created!")

        # Drop tables
        drop_cols = st.columns(2)
        with drop_cols[0]:
            tables_to_drop = st.multiselect("Select tables to drop", [t for t in table_names if t != "data"],
                                          key=f"{section}_tables_to_drop")
            if st.button("❌ Drop Selected Tables", key=f"{section}_drop_tables_btn"):
                for t in tables_to_drop:
                    del sqlq_tables[t]
                session_manager.set_data(section, 'sqlq_tables', sqlq_tables)
                st.success(f"Dropped {len(tables_to_drop)} tables")
                st.rerun()

        # Export all tables
        with drop_cols[1]:
            export_format = st.selectbox("Export format", ["CSV", "Excel"], 
                                       key=f"{section}_export_format")
            if st.button("💾 Export All Tables", key=f"{section}_export_all_btn"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "database_export.zip")
                    with zipfile.ZipFile(zip_path, "w") as zipf:
                        for tname, tdf in sqlq_tables.items():
                            if export_format == "CSV":
                                csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                                zipf.writestr(f"{tname}.csv", csv_bytes)
                            else:
                                excel_buffer = io.BytesIO()
                                tdf.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                                excel_buffer.seek(0)
                                zipf.writestr(f"{tname}.xlsx", excel_buffer.read())

                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="Download All Tables",
                            data=f.read(),
                            file_name="sql_database_export.zip",
                            mime="application/zip",
                            key=f"{section}_download_all_tables"
                        )


def fix_column_case(query, df):
    # Build a mapping of lowercase column names to actual column names
    col_map = {col.lower(): col for col in df.columns}
    # Regex to find words that could be column names (not in quotes)
    def repl(match):
        word = match.group(0)
        return col_map.get(word.lower(), word)
    # Only replace words that are not SQL keywords or numbers
    keywords = set([
        'select', 'from', 'where', 'and', 'or', 'not', 'as', 'on', 'join', 'left', 'right', 'inner', 'outer',
        'group', 'by', 'order', 'limit', 'offset', 'insert', 'into', 'values', 'update', 'set', 'delete',
        'create', 'table', 'drop', 'alter', 'add', 'distinct', 'having', 'union', 'all', 'case', 'when', 'then', 'else', 'end'
    ])
    # Replace only words that are not keywords or numbers
    return re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', lambda m: repl(m) if m.group(0).lower() not in keywords else m.group(0), query)