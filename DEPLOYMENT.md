## Deploying the Containerized Streamlit App on AWS EC2

This section details the steps required to deploy the Dockerized Streamlit application on an AWS EC2 instance.

### Action 1: Launch an EC2 Instance

This step involves creating a virtual server on AWS where we will deploy the dockerized Streamlit application.

1.  **Go to the EC2 Console:**
    *   Navigate to the AWS Management Console using your web browser.
    *   Go to the EC2 section by searching for it or by going to the Services Menu.
    *   The EC2 Dashboard: You'll now be in the EC2 dashboard where you can manage your virtual servers.
2.  **Click "Launch Instance":**
    *   On the EC2 dashboard, find and click the prominent "Launch Instance" button. This will begin the process of creating a new EC2 instance.
3.  **Choose an AMI (Amazon Machine Image):**
    *   An AMI is a pre-configured image that defines the operating system, applications, and other settings for your EC2 instance.
    *   Choose an Ubuntu 20.04 or 22.04 image, as it's a common distribution for web servers. Make sure that the image has the `64-bit (x86)` architecture and not `ARM` as it might not be compatible with all of our libraries.
    *   Select the "Free Tier eligible" option when choosing your AMI, so it is within the limits of the AWS free tier.
4.  **Choose an Instance Type:**
    *   This specifies the amount of compute power (CPU, memory, etc.) for your virtual server.
    *   For a basic Streamlit application, the `t2.micro` instance type is usually sufficient, as it fits within the AWS free tier, but you should check your limits in the AWS web console.
    *   Instance Type Considerations: If you have a more demanding application, you'll need a more powerful (and likely more costly) instance type.
5.  **Configure a Security Group:**
    *   A security group acts as a firewall for your EC2 instance, controlling the traffic that can go in and out of your EC2 instance.
    *   **Inbound Rules:** Create an inbound rule to allow incoming TCP traffic on port `8501`, which is used by default by Streamlit. You can also use the `0.0.0.0/0` source to allow anyone to access your app on the internet, but it should be restricted in production use.
    *   Also, make sure to allow SSH access (port `22`).
    *   Security: Make sure that you create the security rules with the minimum required permissions, in a production application, the use of `0.0.0.0/0` is not recommended.
6.  **Launch the Instance:**
    *   Review all of the previous configurations and launch the instance, your instance will be running in a few minutes.
7.  **Create a Key Pair:**
    *   During the launch process, you'll be prompted to use a key pair. Key pairs are used to securely access the virtual server using SSH.
    *   If you don't have a key pair, create one during this step and download the private key file. It's essential to store this file in a secure location as it is required to connect to the virtual server.

### Action 2: Connect to Your EC2 Instance

After the instance is running, you need to connect to it using SSH.

1.  **Get the Public IP:**
    *   Go to the EC2 Instances page, and look at the IPV4 Public IP for your instance, you'll need it in the next step.
2.  **Connect using SSH:**
    *   Open a terminal or command prompt on your computer.
    *   Use the following command to connect, replacing the values between brackets with your values:
        ```bash
         ssh -i "<your-private-key-path>" ubuntu@<your-public-ip>
        ```
        *    `-i "<your-private-key-path>"`: This tells SSH to use your downloaded private key file for authentication.
        *   `ubuntu@`: The default username in Ubuntu is `ubuntu` but you may use different credentials depending on the image you have launched.
        *   `<your-public-ip>`: This is the public IP of your running instance.
    *   If you are using Windows, and you don't have an SSH client, you can use Putty or a similar tool to connect.

### Action 3: Install Docker and Docker Compose

Once you are connected to your EC2 Instance, you need to install docker.

1.  **Update package lists:**
    *   You need to make sure that the package lists are up-to-date, so run the following command:
        ```bash
        sudo apt update
        ```
2.  **Install Docker**
    *   Install the docker engine:
        ```bash
        sudo apt install docker.io -y
        ```
    * The `-y` tag will accept all prompts and install automatically.
3.  **Enable Docker Service:**
    * Run the following command to enable docker to start automatically on boot.
        ```bash
        sudo systemctl enable docker
        ```
4.  **Start Docker:**
    * Run the following command to start the docker service:
          ```bash
           sudo systemctl start docker
          ```
    * To check that the service is working correctly use:
        ```bash
        sudo systemctl status docker
        ```
    * You should see that the service is `active (running)`.
5.  **Install Docker Compose:**
    *   Download the Docker Compose binary:
        ```bash
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        ```
    *   Make it executable:
        ```bash
        sudo chmod +x /usr/local/bin/docker-compose
        ```
   * To check that the installation worked, you can run the following command:
       ```bash
        docker-compose version
       ```
       You will see a long output with version information for `docker-compose`.

### Action 4: Run the Docker Container on EC2

1.  **Pull the image**
    * Make sure to pull the image before running it.
        ```bash
         docker pull used-car-price-prediction
        ```
2.  **Run the Docker Container:**
    *   Use the `docker run` command to start the container, mapping port 8501:
        ```bash
        docker run -p 8501:8501 used-car-price-prediction
        ```
        *   `-p 8501:8501` maps the container's port 8501 to the EC2 instance's port 8501.

### Action 5: Access Your Application

1.  **Open Your Browser:**
    *   Use your web browser to access the running application using the public IP and port 8501:
        ```
        http://<your-public-ip>:8501
        ```
        *   Replace `<your-public-ip>` with the public IP of your EC2.
   *    Your Streamlit application should be working as expected.