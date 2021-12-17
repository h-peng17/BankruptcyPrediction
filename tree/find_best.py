import os

for year in [1, 2, 3, 4, 5]:
    for model in ['tree', 'forest']:
        best = 0
        best_dev = 0
        best_config = None
        try:
            for config in os.listdir('./log/year%d/%s' % (year, model)):
                with open(os.path.join('./log/year%d/%s' % (year, model), config, 'result.txt'), 'r') as f:
                    lines = f.readlines()
                    if 'Dev result. F1:' in lines[1]:
                        f1 = float(lines[1].split()[3][:-1])
                        if f1 > best_dev:
                            best_dev = f1
                            best_config = config
                            best = lines[-1].strip()
        except FileNotFoundError:
            continue
        print(year)
        print(model)
        print(best)
        print(best_config)
