# Real-Robot Upload

Deploying a PyTorch (.pt) reinforcement learning policy onto the Unitree G1 EDU (23 DoF) involves a strict, step-by-step process. You must configure your local PC, establish an SSH link, copy the files, and run the model via the onboard Jetson Orin computing unit. [1, 2] 
Follow these steps carefully to transfer and launch your policy safely.
------------------------------
## Step 1: Physical Setup & Safety

   1. Hang the robot: Suspend the G1 EDU completely off the ground using its protective suspension bracket. Do not test new policies on the floor.
   2. Connect the cable: Plug an Ethernet cable from your local Ubuntu PC into physical Port 4 or Port 5 on the G1's internal network panel. [3, 4, 5] 

## Step 2: Configure Your Local PC Network
Your computer must share the same local network subnet as the robot. [5, 6] 

   1. Open your Ubuntu Network Settings. [2] 
   2. Select your Wired connection and change the IPv4 method to Manual. [2] 
   3. Set your parameters exactly as follows:
   * IP Address: 192.168.123.51 (or any free IP between .50 and .100)
      * Netmask: 255.255.255.0 (or 24)
      * Gateway: Leave blank or use 192.168.123.1 [2, 5, 7] 
   4. Apply the settings and restart your wired network connection. [2] 

Test the connection by opening your terminal and typing:

ping 192.168.123.164

(If you get a response, your PC can see the G1's secondary development computer). [2] 
## Step 3: Copy the .pt Policy Using SCP
Before logging into the robot, transfer your policy file from your local computer's download folder to the robot's onboard Jetson module using Secure Copy (scp). [8] 
Open a terminal on your host PC and run:

scp /path/to/your/policy_23dof.pt unitree@192.168.123.164:~/unitree_rl_gym/models/

When prompted for the password, type 123 and press Enter. [2, 9] 
## Step 4: SSH Into the Robot
Now, log directly into the G1's Jetson Orin system to configure and handle execution: [1, 2] 

ssh -X unitree@192.168.123.164

Enter the default password: 123 [2] 
Once inside, navigate to your policy framework directory (for example, the standard Unitree RL framework):

cd ~/unitree_rl_gym

## Step 5: Verify the 23 DoF Mapping
Most open-source code repositories default to a 29 DoF configuration (which accounts for extra wrist and waist joints). Since you have the 23 DoF model, you must edit your local environment configuration file before executing the policy. [10] 

   1. Open your model's configuration file (usually config.yaml or a joint mapping python file) using the text editor:
   
   nano config/g1_config.py
   
   2. Look for the num_actions or dof_names array. Ensure it is set to 23 and does not include the 6 wrist/waist control dimensions found on the 29 DoF version. Save and close (Ctrl+O, then Ctrl+X). [11] 

## Step 6: Execute Policy Inference
With the file copied and configured, initialize your python runtime environment (typically managed via Conda on the G1): [1] 

# Activate your local deployment environment
conda activate g1_env
# Run the deployment tracking script pointing to your transferred weights
python scripts/deploy_policy.py --model_path ./models/policy_23dof.pt --robot_dof 23

The script will begin feeding observation data to your .pt model and returning target motor positions to the main locomotion controller. Keep a hand near the robot's hardware emergency stop switch or your keyboard (Ctrl+C) during the initial startup to stop execution if the motors shake or loop incorrectly. [1, 12] 
Are you getting any network timeout errors when pinging the robot, or do you need a template python script snippet to see how the observation vector maps to the 23 DoF joint layouts?

[1] [https://robotics.stackexchange.com](https://robotics.stackexchange.com/questions/115398/deploy-policy-to-unitree-g1-on-jetson-orin-nx)
[2] [https://docs.quadruped.de](https://docs.quadruped.de/projects/g1/html/g1_ros2_driver.html)
[3] [https://support.unitree.com](https://support.unitree.com/home/en/G1_developer/Operational_guidance)
[4] [https://docs.westonrobot.com](https://docs.westonrobot.com/tutorial/unitree/g1_internet_guide/)
[5] [https://docs.westonrobot.com](https://docs.westonrobot.com/tutorial/unitree/g1_dev_guide/)
[6] [https://support.unitree.com](https://support.unitree.com/home/en/developer/module_update)
[7] [https://support.unitree.com](https://support.unitree.com/home/en/developer/SLAM%20and%20Navigation_service)
[8] [https://www.youtube.com](https://www.youtube.com/watch?v=9dW3TPlKlVw)
[9] [https://support.unitree.com](https://support.unitree.com/home/en/developer/module_update)
[10] [https://www.linkedin.com](https://www.linkedin.com/posts/robostore-llc_humanoidrobotics-unitreeg1-roboticseducation-activity-7424911365875712000-JvxS)
[11] [https://docs.flowiseai.com](https://docs.flowiseai.com/configuration/deployment/digital-ocean)
[12] [https://tallyfy.com](https://tallyfy.com/products/pro/integrations/robotics/unitree/)

