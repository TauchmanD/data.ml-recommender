from Recommender.group_aggregation_pipeline import aggregate_with_average, aggregate_with_least_misery, aggregate_with_custom_method
from Recommender.recommender import Recommender
from Recommender.sequence_pipeline import main as sequence_run
from rich_menu import Menu
from rich.spinner import Spinner
from rich.console import Console


def menu():
    menu = Menu(
        "Compute prediction for userId 3 and movie 3",
        "Compute predictions for userId 3 and movie 3, but using custom similarity function (Popularity-Discounted, Significance-Weighted Pearson similarity.)",
        "Compute predictions for 5 random users that form a group, aggregated using AVERAGE",
        "Compute predictions for 5 random users that form a group, aggregated using LEAST_MISERY",
        "Compute predictions for 5 random users that form a group, aggregated using Custom aggregation",
        "Compute sequential predictions with our proposed method",
        "Exit",
        panel_title="Recommender Menu",
        selection_char="->",
    )
    console = Console()
    match menu.ask():
        case "Compute prediction for userId 3 and movie 3":
            r = Recommender.load_from_path("ml-latest-small/ratings.csv")
            print(r.predict(3, 3))
        case "Compute predictions for userId 3 and movie 3, but using custom similarity function (Popularity-Discounted, Significance-Weighted Pearson similarity.)":
            r = Recommender.load_from_path("ml-latest-small/ratings.csv", sim_func="PD_SignificanceWeightedPearson")
            print(r.predict(3, 3))
        case "Compute predictions for 5 random users that form a group, aggregated using AVERAGE":
            with console.status("[bold green] Computing...", spinner="dots"):
                aggregate_with_average()
        case "Compute predictions for 5 random users that form a group, aggregated using LEAST_MISERY":
            with console.status("[bold green] Computing...", spinner="dots"):
                aggregate_with_least_misery()
        case "Compute predictions for 5 random users that form a group, aggregated using Custom aggregation":
            with console.status("[bold green] Computing...", spinner="dots"):
                aggregate_with_custom_method()
        case "Compute sequential predictions with our proposed method":
            with console.status("[bold green] Computing...", spinner="dots"):
                sequence_run()
        case "Exit":
            exit()

def main():
    menu()

if __name__ == "__main__":
    main()
