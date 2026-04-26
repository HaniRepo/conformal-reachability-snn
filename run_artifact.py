import subprocess
import sys


MENU = {
    "1": ("CWRU case study", "analyze_cwru.py"),
    "2": ("Filtration case study", "analyze_filter.py"),
    "3": ("NASA FD001 case study", "analyze_nasa.py"),
    "4": ("NASA fusion/extra comparison", "analyze_fusion.py"),
    "5": ("Extended filtration analysis", "analyze_extended_filter.py"),
}


def run_script(script_name: str):
    print("\n" + "=" * 70)
    print(f"Running: {script_name}")
    print("=" * 70 + "\n")

    result = subprocess.run([sys.executable, script_name])

    if result.returncode == 0:
        print("\n" + "-" * 70)
        print(f"Finished successfully: {script_name}")
        print("-" * 70)
    else:
        print("\n" + "!" * 70)
        print(f"Error while running: {script_name}")
        print(f"Return code: {result.returncode}")
        print("!" * 70)


def show_menu():
    print("\n" + "=" * 70)
    print("Artifact Runner")
    print("=" * 70)
    print("If you would like to check the CWRU case study, press 1")
    print("If you would like to check the filtration case study, press 2")
    print("If you would like to check the NASA FD001 case study, press 3")
    print("If you would like to check the NASA fusion/comparison study, press 4")
    print("If you would like to check the extended filtration analysis, press 5")
    print("If you want to quit, press 0")
    print("=" * 70)


def main():
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Exiting artifact runner.")
            break

        if choice not in MENU:
            print("Invalid choice. Please enter 0, 1, 2, 3, 4, or 5.")
            continue

        title, script_name = MENU[choice]
        print(f"\nSelected: {title}")
        run_script(script_name)

        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()