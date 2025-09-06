import subprocess

def run_pipeline():
    # Lancer la stack docker-compose
    subprocess.run(["docker-compose", "up", "--build", "-d"])

    # Optionnel : attendre que collector termine puis lancer prepare
    # (mais ça peut être géré par depends_on dans docker-compose)
    print("All services are up and running.")

if __name__ == "__main__":
    run_pipeline()
