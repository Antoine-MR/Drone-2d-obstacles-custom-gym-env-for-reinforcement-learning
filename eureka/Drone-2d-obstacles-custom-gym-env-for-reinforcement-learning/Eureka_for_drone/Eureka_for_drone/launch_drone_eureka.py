import os
import subprocess

def main():    
    if not os.path.exists("eureka/eureka.py"):
        print("Error: This script must be executed in the Eureka directory")
        return
        
    cmd = [
        "python", "eureka/eureka.py",
        "env=drone_2d",
        "iteration=5",        
        "sample=2",           
        "max_iterations=1000",
        "model=llama3.1:8b",
        "suffix=drone"        
    ]
    
    print("Executing:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=False)
        print("Eureka execution completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error during Eureka execution: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
if __name__ == "__main__":
    main()