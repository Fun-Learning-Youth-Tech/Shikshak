# Teacher-AI (Shikshak)
With Cerebras’ Fast Inferface with Claude API, we built a Teaching and Mentoring Chatbot, Shikshak, that will curate suggestions and responses based on each student’s individual profiles. It is multilingual, prompts small step by step tasks encouraging students to think, and can work in low-internet conditions. 

![FLY Image](<Screenshot 2024-11-06 at 2.25.59.png>)

We built this app with Python on backend and JavaScipt on frontend. Since this was a Proof of Concept, we built a basic multilingual chatbot with Cerebras Inferface being called with every user prompt. User has to enter their ID to begin the conversation and gets personalized recommendations on what to learn and how to learn it based on their individual profile which will be studied over time. we have used AI to generate code based on my needs for faster building process and have debugged and edited the code based on my own judgement. It hasn’t been tested on students yet, but I want to soon deploy it and make it publicly available through Chrome or provide it as a software to our teachers so it can reach our students in India.

We called the Cerebras API key and used a RAG model to refine the algorithm. We then stored user data after removing names and personal information and replacing them with their student IDs. The plan is to keep enough information to curate Shikshak so it can best teach students in a way they can understand and access information from the world. It keeps students engaged with interactive problem solving while maintaining anonymity.

Cerebras Fast Inference helps me solve the problem of minimal connectivity and multilingual input and output. It helps me load student data and create student profiles based on that. The biggest problems in this projects are effectively solved by Cerebras’ Fast Inference which acts as a bridge between the vast information on the internet and underprivileged students in developing countries.

