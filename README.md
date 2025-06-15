🚗 EV Charge Scheduling using Deep Q-Learning
This project implements a reinforcement learning approach for intelligent scheduling of Electric Vehicle (EV) charging using Deep Q-Learning (DQN). The goal is to optimize EV battery charging and discharging decisions based on real-world data including electricity prices, household demand, battery state of charge (SoC), and car availability.

🔍 Key Features
Custom EV Environment: Simulates realistic EV charging scenarios with constraints like SoC levels and car availability.

Deep Q-Network (DQN): Trains an agent to minimize energy costs while ensuring daily driving needs are met.

Real-World Dataset Integration: Supports time-series CSV input with columns like Month, Day, Hour, Price, Household Demand, SoC, and Car_Available.

Smart Energy Decisions: Learns optimal policies to charge, discharge, or stay idle based on dynamic inputs.

Detailed Logging & Evaluation: Logs episode-wise performance, plots training reward trends, and compares AI vs baseline cost savings.

📁 Dataset Columns
Month, Day, Hour

Price: Grid electricity price

Household_Demand: Energy demand of home

SoC: Battery State of Charge

Car_Available: Whether the EV is at home or away

🧠 Objective
Reduce electricity cost and manage EV battery efficiently under real-time constraints, enabling smarter home energy systems and better load balancing.

