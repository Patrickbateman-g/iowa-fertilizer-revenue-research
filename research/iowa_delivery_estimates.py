import argparse
import csv
import io
import zipfile
from pathlib import Path
from statistics import mean, median


def read_rows(directory, filename):
    path = directory / filename
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))
    with zipfile.ZipFile(directory / "iowa-nitrogen-volume.zip") as bundle:
        data = bundle.read("iowa-nitrogen-volume/" + filename)
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def slope(x, y):
    x_mean, y_mean = mean(x), mean(y)
    return sum((a-x_mean)*(b-y_mean) for a,b in zip(x,y)) / sum((a-x_mean)**2 for a in x)


def backtest(directory):
    spring = sorted(read_rows(directory, "paired.csv"), key=lambda r: int(r["year"]))
    fall = read_rows(directory, "fall_tonnage.csv")
    fall_by_year = {int(r["spring_year"]): r for r in fall}
    years = [int(r["year"]) for r in spring]
    if years != list(range(2014,2025)) or len(fall) != len(years) or set(fall_by_year) != set(years):
        raise ValueError("Expected a matched pair of seasonal records for each year from 2014 to 2024.")
    volume = [float(r["nitrogen_tons"]) for r in spring]
    rain = [float(r["rain_mm"]) for r in spring]
    autumn = [sum(int(fall_by_year[y][f"fall_grade{g}_product_tons"])*g/100
                  for g in (82,46,32,28)) for y in years]

    output = []
    for i in range(5,len(years)):
        recent = slice(i-3,i)
        beta = slope(rain[:i], volume[:i])
        dx = [rain[j]-rain[j-1] for j in range(1,i)]
        dy = [volume[j]-volume[j-1] for j in range(1,i)]
        change_slope = sum(a*b for a,b in zip(dx,dy)) / sum(a*a for a in dx)
        recent_volume = mean(volume[recent])
        recent_median = median(volume[recent])
        rain_adjustment = beta * (rain[i] - mean(rain[recent]))
        rain_estimate = recent_median + rain_adjustment
        bounded_rain_estimate = min(max(rain_estimate, min(volume[recent])), max(volume[recent]))
        recent_crop_total = mean(volume[j]+autumn[j] for j in range(i-3,i))
        predictions = {
            "All earlier springs": mean(volume[:i]),
            "Original rain model": mean(volume[:i])+beta*(rain[i]-mean(rain[:i])),
            "Previous spring": volume[i-1],
            "Previous spring + rain change": volume[i-1]+change_slope*(rain[i]-rain[i-1]),
            "Three recent springs": recent_volume,
            "Median of three recent springs": recent_median,
            "Three-spring median + rain": rain_estimate,
            "Three-spring median + bounded rain": bounded_rain_estimate,
            "Three recent springs + rain": recent_volume + rain_adjustment,
            "Three crop years minus prior fall": max(0,recent_crop_total-autumn[i]),
        }
        output.append((years[i],volume[i],predictions))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest Iowa spring nitrogen deliveries.')
    parser.add_argument("--annual",action="store_true",help="also print predictions for each year")
    args = parser.parse_args()
    rows = backtest(Path(__file__).resolve().parent)
    print("Model | mean absolute error in nitrogen tons, 2019–2024")
    for model in rows[0][2]:
        print(f"{model} | {mean(abs(p[model]-actual) for _,actual,p in rows):,.0f}")
    if args.annual:
        for year,actual,models in rows:
            print(f"\n{year} reported: {actual:,.0f} tons")
            for model,prediction in models.items():
                print(f"  {model}: {prediction:,.0f}")
