# 1. Start from a base image that already has Python installed
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the dependency list first, and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Now copy the rest of your actual code
COPY . .

# 5. The command that runs when the container starts
CMD ["adk", "api_server", "--port", "8080"]