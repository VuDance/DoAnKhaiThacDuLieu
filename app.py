import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder

st.title("🛒 Demo Apriori Algorithm")

# Upload file
uploaded_file = st.file_uploader("Upload dataset (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df = None
    try:
        # Thử đọc bằng Excel
        df = pd.read_excel(uploaded_file, header=None, engine="openpyxl")
    except:
        try:
            # Thử đọc như CSV
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
        except:
            try:
                # Đọc thẳng như text
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, sep="\t")
            except Exception as e:
                st.error(f"❌ Không thể đọc file: {e}")
                st.stop()

    st.write("📊 Preview dữ liệu:")
    st.dataframe(df.head())

    # Chuyển dữ liệu thành list transaction
    transactions = df.stack().groupby(level=0).apply(list).tolist()

    # One-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    dataset = pd.DataFrame(te_ary, columns=te.columns_)

    # Nhập tham số
    min_support = st.slider("Min support", 0.01, 0.5, 0.02, 0.01)
    min_confidence = st.slider("Min confidence", 0.1, 1.0, 0.5, 0.05)

    # Chạy Apriori
    frequent_itemsets = apriori(dataset, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Chuyển frozenset sang list để Streamlit hiển thị không lỗi Arrow
    frequent_itemsets_display = frequent_itemsets.copy()
    frequent_itemsets_display["itemsets"] = frequent_itemsets_display["itemsets"].apply(lambda x: list(x))

    rules_display = rules.copy()
    rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: list(x))
    rules_display["consequents"] = rules_display["consequents"].apply(lambda x: list(x))

    st.subheader("📝 Glossary / Notes on Terms")

    st.markdown("""
    - **Support**: Tỷ lệ transaction chứa itemset.  
        - Ví dụ: Support = 0.2 → 20% transaction có itemset này.  

    - **Confidence**: Xác suất xuất hiện Y khi X xuất hiện.  
        - Confidence cao → luật chắc chắn hơn.  

    - **Lift**: Đo mức độ tăng khả năng xuất hiện của Y khi X xuất hiện, so với khi X và Y xuất hiện độc lập.  
        - Lift > 1 → X làm tăng khả năng xảy ra của Y (luật hữu ích).  
        - Lift = 1 → X không ảnh hưởng đến Y.  
        - Lift < 1 → X làm giảm khả năng xảy ra của Y (luật không hữu ích).  

    - **Antecedents**: Item hoặc tập item bên trái của luật (X trong X → Y).  
    - **Consequents**: Item hoặc tập item bên phải của luật (Y trong X → Y).  
    - **Itemset**: Một tập hợp các item xuất hiện cùng nhau trong transaction.  
    """)


    # Hiển thị
    st.subheader("✅ Frequent Itemsets")
    st.dataframe(frequent_itemsets_display.sort_values("support", ascending=False).head(1000))

    st.subheader("✅ Association Rules")
    rules_display = rules.copy()
    rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ', '.join(list(x)))
    rules_display = rules_display[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules_display = rules_display[["antecedents", "consequents", "support", "confidence", "lift"]]
    st.dataframe(rules_display.head(1000))

    # # Visualization: Top items by support
    # st.subheader("📈 Top Items by Support")
    # top_items = frequent_itemsets.nlargest(20, "support")
    # fig, ax = plt.subplots()
    # ax.barh([str(i) for i in top_items["itemsets"]], top_items["support"])
    # ax.set_xlabel("Support")
    # ax.set_ylabel("Itemsets")
    # st.pyplot(fig)

    # Visualization: Scatter Confidence vs Lift
    st.subheader("📈 Scatter: Confidence vs Lift")
    if not rules.empty:
        fig2, ax2 = plt.subplots()
        ax2.scatter(
            rules["confidence"].astype(float),
            rules["lift"].astype(float),
            s=(rules["support"].astype(float) * 2000),
            alpha=0.5
        )
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Lift")
        st.pyplot(fig2)
    else:
        st.info("Không có luật nào thỏa mãn tham số hiện tại.")
