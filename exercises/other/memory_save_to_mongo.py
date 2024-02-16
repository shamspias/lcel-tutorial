from langchain_community.chat_message_histories import MongoDBChatMessageHistory

chat_message_history = MongoDBChatMessageHistory(
    session_id="test_session1",
    connection_string="mongodb://localhost:27017",
    database_name="legal_data",
    collection_name="chat_histories",
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")
