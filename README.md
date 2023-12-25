# Content-Moderation-as-a-Service


## Project Goals: The primary goal of our project is to implement a scalable service-oriented architecture for Content Moderation System  
capable of analyzing text inputs and URLs provided by users and predicting multiple labels, including toxic, severe toxic, obscene, insult, threat, and identity_hate. The project aims to integrate machine learning models for accurate label prediction and provide scalable solution.

Architecture Diagram:



Components Overview :

Rest Client/Frontend:

The REST client can be any client with access to our servers. 
Currently the rest clients are exposed using GKE Ingress based load balancers with no authentication in place.
This is done with the assumption that only administrators should be able to access such a service dealing with sensitive content but is also a limitation that we have not tackled.

Load Balancer:

The purpose of load balancing is to serve as the gateway to our cluster. It balances the load on front end server deployment as the frontend can receive a high volume of requests from users.
As mentioned previously, we are using GKE Ingress in Google Cloud Platform for load balancing and Nginx for local testing purposes.

REST Server/Frontend

This is the main kubernetes deployment which receives user requests for processing content.
We have deployed 3 replicas for demo purposes but also added logs for when requests are beyond a certain threshold to increase promptly.
Internally the frontend is a Flask application supporting the following APIs:
/api/predict - POST Method is used for real time data prediction for small texts.
/api/upload - POST Method is used for uploading data to frontend and classifying the document.
/api/fileresult - GET Method is used for getting results back of large files.
The frontend processes small text data in real time and returns the response back to users.
For larger files we use Redis as a message queue to populate “queue” and let an independent deployment of worker nodes handle them.

Redis

Redis is used primarily for serving as a message queue in this application. It maintains the worker request queue for large files which are stored in object storage.
The file is represented by a unique UUID as file name in storage and the hash is pushed to them queue.
A redis queue “logging” is also used for consolidating all logs from different pods generated during the end to end workflow.
A limitation is that Redis may be slower in heavy workloads and more dedicated message queue service such as RabbitMQ may be better.
We have also used redis as a cache to maintain frequently requested text and file result easily accessible.

MinIO/Data Storage

While our application is a compute heavy application, we do need object storage for large files. 
MinIO has been used for this application but its scaling management gets hard and managed cloud storage such as S3 and Google Cloud’s object storage is a much easier to integrate and maintain.

Model Execution:

Utilizes DistilBERT for pretraining.
Predicts multiple labels for content moderation.
Enables identification of toxic elements in text.

ML Worker

ML Worker is responsible for processing large files that are too expensive to be processed in real time at frontend.
The worker will load text files from “queue” bucket in Minio for hash populated in “queue” Redis List as message queue.
After processing the file, the worker node populates the results along with the file in “output” bucket in minio.
A limitation of this deployment is that we are using Minio for this project but its scaling gets hard. A potential solution is to move to a managed object storage such as google cloud object storage or S3.

Backend Infrastructure:

Implemented using Docker and Kubernetes for containerization and orchestration.
Flask serves as the backend web-server API for handling requests.
Nginx Ingress Controller facilitates load balancing and port forwarding in local deployment and GCE Ingress in GCP.

REST Client Technologies (Web interface):

Developed with HTML, CSS, and Angular for an interactive user interface.
Facilitates user input and interaction with the Flask backend.



Caching and Messaging:

Redis key-value store used for both caching and messaging queues.
Worker queue for queuing incoming requests.
Logging queue for recording information during the process.

Debugging and Logging:

Incorporates print statements and log_info, log_debug for effective debugging.
Logs essential information for troubleshooting and system monitoring.
We maintain a separate deployment “logs” to collect all logs from different pods for troubleshooting.

Local Deployment:

Minikube employed locally for deploying the Kubernetes cluster.
Docker is utilized for building images and packaging components within containers.

Capabilities and Limits of the Final Solution:

The final CMaaS solution offers several capabilities:
Efficient Content Moderation: The custom ML model enables accurate and efficient moderation of diverse content types.
Scalability: The Dockerized and Kubernetes-managed infrastructure allows for seamless scaling of the CMaaS based on demand.
User-Friendly Interface: The HTML and JavaScript frontend provides a user-friendly interface for submitting content and receiving moderation feedback.
Optimized Performance: Caching with Redis and asynchronous processing through the messaging queue contribute to optimized system performance.

However, the solution also has its limits:

Model Limitations: The ML model's accuracy may be affected by biases in the training data, requiring continuous monitoring and updates.
Frontend Complexity: The frontend may face challenges in handling complex user interactions, necessitating ongoing improvements.
Infrastructure Scaling: While Kubernetes facilitates scaling, optimal configuration and load balancing strategies are crucial to handle extremely high traffic loads effectively.
Data Querying in Object Storage: MinIO's limitations in complex querying may pose challenges in efficiently retrieving specific data from stored files.
