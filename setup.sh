mkdir -p ~/.streamlit/
echo "\
[theme]
primaryColor="#328cc1"
backgroundColor="#ffffff"
secondaryBackgroundColor="#e5e5e5"
textColor="#000000"
font="sans serif"
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
