RU_TEXT = {
    "Version": "Версия",
    "Neuroevolution": "Нейроэволюция",
    "TRAINING": "ОБУЧЕНИЕ",
    "HARDWARE": "ОБОРУДОВАНИЕ",
    "Generation": "Поколение",
    "Current best": "Лучший сейчас",
    "All-time best": "Лучший за всё время",
    "Best score": "Лучший счёт",
    "Best lines": "Лучшие линии",
    "Evaluated": "Проверено",
    "Rotated moves": "Ходы с вращением",
    "CPU Load": "Загрузка CPU",
    "RAM Load": "Загрузка RAM",
    "GPU Load": "Загрузка GPU",
    "GPU Temp": "Температура GPU",
    "Neural VRAM": "VRAM нейросети",
    "Reserved VRAM": "Резерв VRAM",
    "Total VRAM": "Общая VRAM",
    "VRAM LIMIT": "ЛИМИТ VRAM",
    "AGENTS PER GENERATION": "АГЕНТОВ В ПОКОЛЕНИИ",
    "Device": "Устройство",
    "Status": "Состояние",
    "START": "СТАРТ",
    "PAUSE": "ПАУЗА",
    "RESET": "СБРОС",
    "CONFIRM RESET": "ТОЧНО?",
    "SETTINGS": "НАСТРОЙКИ",
    "EXIT": "ВЫХОД",
    "APPLY": "ОК",
    "AUTO": "АВТО",
    "N/A": "Н/Д",
    "Applying fully resets training": "Применение полностью сбросит обучение",
    "MODEL SETTINGS": "НАСТРОЙКИ МОДЕЛИ",
    "Every value is editable. Applying settings starts training from generation 0.": "Все параметры доступны для изменения. Применение начнёт обучение с поколения 0.",
    "Changes are stored locally in .runtime/data/settings.json": "Изменения сохраняются локально в .runtime/data/settings.json",
    "CANCEL": "ОТМЕНА",
    "APPLY & RESET": "ПРИМЕНИТЬ И СБРОСИТЬ",
    "APPLY CHANGES": "ПРИМЕНИТЬ",
    "DEFAULTS": "ПО УМОЛЧАНИЮ",
    "Press CONFIRM RESET to erase the current training and apply all values": "Нажмите ПОДТВЕРДИТЬ, чтобы удалить текущее обучение и применить все значения",
    "Default model values loaded. Press APPLY & RESET to use them.": "Значения модели по умолчанию загружены. Нажмите ПРИМЕНИТЬ И СБРОСИТЬ.",
    "Automatic values loaded for: {profile}. Press APPLY & RESET to use them.": "Автоматические значения подобраны для: {profile}. Нажмите ПРИМЕНИТЬ И СБРОСИТЬ.",
    "Changing only the language keeps the current checkpoint.": "Смена только языка сохранит текущий checkpoint.",
    "LANGUAGE": "ЯЗЫК",
    "GAME AND NETWORK": "ИГРА И НЕЙРОСЕТЬ",
    "EVOLUTION": "ЭВОЛЮЦИЯ",
    "RULE-AWARE PLACEMENT": "РАЗМЕЩЕНИЕ ПО ПРАВИЛАМ",
    "GENERATION FITNESS": "ОЦЕНКА ПОКОЛЕНИЯ",
    "EVALUATION": "ВЫЧИСЛЕНИЯ",
    "CUDA AND MEMORY": "CUDA И ПАМЯТЬ",
    "INTERFACE AND MONITORING": "ИНТЕРФЕЙС И МОНИТОРИНГ",
    "Language": "Язык",
    "Interface language. Click the field to switch between Russian and English; the choice is saved after applying.": "Язык всего интерфейса. Нажмите на поле для переключения русского и английского; выбор сохранится после применения.",
    "Board width": "Ширина поля",
    "Number of columns in the Tetris board. The network input and output layers are rebuilt when this changes.": "Количество колонок поля Tetris. При изменении перестраиваются входной и выходной слои сети.",
    "Board height": "Высота поля",
    "Number of visible rows in the Tetris board. A taller board increases the network input size and GPU memory use.": "Количество видимых строк поля Tetris. Более высокое поле увеличивает вход сети и расход памяти GPU.",
    "Agents per generation": "Агентов в поколении",
    "Number of independently evaluated neural-network agents in one generation. More agents improve search diversity but require more memory and time.": "Количество независимо оцениваемых нейросетевых агентов. Больше агентов повышает разнообразие поиска, но требует больше памяти и времени.",
    "Hidden layer sizes": "Размеры скрытых слоёв",
    "Comma-separated neuron counts for every hidden layer, for example 96, 64. This controls model capacity and memory use.": "Количество нейронов каждого скрытого слоя через запятую, например 96, 64. Определяет ёмкость модели и расход памяти.",
    "Elite agents": "Элитных агентов",
    "Number of highest-fitness agents copied unchanged into the next generation.": "Количество лучших по fitness агентов, которые без изменений копируются в следующее поколение.",
    "Parent pool": "Родительский пул",
    "Number of highest-fitness agents eligible to become parents. It must be at least the elite count.": "Количество лучших агентов, доступных для создания потомков. Не может быть меньше элиты.",
    "Mutation probability": "Вероятность мутации",
    "Probability that each inherited neural-network parameter is mutated. Range: 0 to 1.": "Вероятность мутации каждого унаследованного параметра сети. Диапазон: от 0 до 1.",
    "Mutation strength": "Сила мутации",
    "Standard deviation of random changes applied to mutated parameters. This acts like evolutionary temperature: higher values explore more aggressively.": "Разброс случайных изменений мутировавших параметров. Работает как температура эволюции: высокое значение усиливает исследование.",
    "Crossover probability": "Вероятность скрещивания",
    "Probability that a child combines parameters from two parents instead of inheriting one parent. Range: 0 to 1.": "Вероятность объединения параметров двух родителей вместо наследования одного. Диапазон: от 0 до 1.",
    "Neural policy influence": "Влияние нейросети",
    "Multiplier for the neural network score when it is combined with the rule-aware placement score.": "Множитель оценки нейросети при объединении с оценкой размещения по правилам.",
    "Random seed": "Случайный seed",
    "Base seed for initial weights and deterministic piece sequences. The same seed and settings make experiments reproducible.": "Базовый seed начальных весов и последовательностей фигур. Одинаковые seed и настройки делают эксперименты воспроизводимыми.",
    "Line reward": "Награда за линии",
    "Immediate placement bonus for completed lines before the neural policy score is added.": "Мгновенная награда за заполненные линии до добавления оценки нейросети.",
    "Height penalty": "Штраф за высоту",
    "Immediate penalty for the sum of all column heights. Higher values favor flatter, lower stacks.": "Мгновенный штраф за сумму высот колонок. Высокое значение поощряет низкий ровный стек.",
    "Hole penalty": "Штраф за дыры",
    "Immediate penalty for empty cells trapped below blocks. Higher values strongly discourage inaccessible gaps.": "Мгновенный штраф за пустые клетки под блоками. Высокое значение сильнее запрещает недоступные промежутки.",
    "Bumpiness penalty": "Штраф за неровность",
    "Immediate penalty for height differences between adjacent columns.": "Мгновенный штраф за перепады высоты соседних колонок.",
    "Peak-height penalty": "Штраф за пик высоты",
    "Immediate penalty for the tallest column, reducing the risk of an early game over.": "Мгновенный штраф за самую высокую колонку, снижающий риск раннего Game Over.",
    "Lines fitness reward": "Fitness-награда за линии",
    "Final fitness reward per cleared line. This is the strongest signal for learning valid Tetris play.": "Итоговая fitness-награда за каждую очищенную линию. Главный сигнал для обучения правильной игре.",
    "Survival reward": "Награда за выживание",
    "Final fitness reward per successfully placed piece. It rewards survival without replacing the line-clear objective.": "Итоговая награда за каждую размещённую фигуру. Поощряет выживание, не заменяя цель очистки линий.",
    "Height fitness penalty": "Fitness-штраф за высоту",
    "Final fitness penalty for total column height at the end of an agent's game.": "Итоговый fitness-штраф за суммарную высоту колонок в конце игры агента.",
    "Hole fitness penalty": "Fitness-штраф за дыры",
    "Final fitness penalty for holes remaining in the board.": "Итоговый fitness-штраф за оставшиеся на поле дыры.",
    "Bumpiness fitness penalty": "Fitness-штраф за неровность",
    "Final fitness penalty for an uneven board surface.": "Итоговый fitness-штраф за неровную поверхность поля.",
    "Peak fitness penalty": "Fitness-штраф за пик",
    "Final fitness penalty for the highest column.": "Итоговый fitness-штраф за самую высокую колонку.",
    "Game-over penalty": "Штраф за Game Over",
    "Final penalty applied when the agent reaches game over.": "Итоговый штраф при достижении агентом Game Over.",
    "Piece limit per game": "Лимит фигур на игру",
    "Maximum number of pieces evaluated for each agent in one generation. Larger values improve long-game measurement but take longer.": "Максимум фигур для каждого агента за поколение. Большое значение точнее оценивает долгую игру, но требует больше времени.",
    "Evaluation chunk": "Пакет вычислений",
    "Preferred number of agents processed together. CUDA automatically reduces this value if memory is insufficient.": "Желаемое количество одновременно обрабатываемых агентов. CUDA автоматически уменьшает его при нехватке памяти.",
    "Checkpoint interval": "Интервал checkpoint",
    "Number of completed generations between automatic checkpoint saves.": "Количество завершённых поколений между автоматическими сохранениями checkpoint.",
    "Neural VRAM limit": "Лимит VRAM нейросети",
    "Maximum MiB available to neural-network tensors. Set 0 for automatic detection.": "Максимум MiB для тензоров нейросети. Значение 0 включает автоматическое определение.",
    "Automatic VRAM fraction": "Автоматическая доля VRAM",
    "Fraction of total VRAM the automatic limiter may use after reserving memory for the desktop and other applications. Range: 0.05 to 1.": "Доля общей VRAM для автоматического лимита после резерва под систему и другие приложения. Диапазон: от 0,05 до 1.",
    "Minimum VRAM limit": "Минимальный лимит VRAM",
    "Preferred lower bound in MiB for the automatic CUDA memory budget when the GPU has enough free capacity.": "Желаемая нижняя граница автоматического бюджета CUDA в MiB при достаточном объёме GPU.",
    "GPU reserve": "Резерв GPU",
    "MiB kept outside the neural budget for the display, driver, and other applications.": "MiB вне бюджета нейросети для экрана, драйвера и других приложений.",
    "Interface FPS": "FPS интерфейса",
    "Maximum interface refresh rate. It does not change training speed directly.": "Максимальная частота обновления интерфейса. Напрямую не меняет скорость обучения.",
    "Demo drop interval": "Интервал падения в демо",
    "Seconds between visible falling-piece steps in the demonstration board.": "Секунды между видимыми шагами падения фигуры на демонстрационном поле.",
    "Hardware polling interval": "Интервал опроса датчиков",
    "Seconds between CPU, RAM, GPU, temperature, and VRAM measurements. The minimum is 1.5 seconds.": "Секунды между измерениями CPU, RAM, GPU, температуры и VRAM. Минимум — 1,5 секунды.",
    "Number of the generation currently being evaluated. One generation tests every agent and creates the next population.": "Номер оцениваемого поколения. Одно поколение проверяет всех агентов и создаёт следующую популяцию.",
    "Highest fitness reached by an agent in the current generation.": "Максимальный fitness агента в текущем поколении.",
    "Highest fitness reached since the current training run began.": "Максимальный fitness с начала текущего обучения.",
    "Largest standard Tetris score reached by the all-time best agent.": "Наибольший стандартный счёт Tetris у лучшего агента.",
    "Largest line-clear count reached by the all-time best agent.": "Наибольшее количество очищенных линий у лучшего агента.",
    "Agents already finished or eliminated in the current generation versus the total population.": "Завершившие игру или выбывшие агенты текущего поколения относительно всей популяции.",
    "Percentage of placements that use a non-default rotation. It helps confirm that rotations are being explored.": "Доля размещений с поворотом. Помогает убедиться, что модель исследует вращения.",
    "Current total processor utilization. Interface, monitoring, checkpoint work, and CPU-only training contribute to it.": "Текущая общая загрузка процессора интерфейсом, мониторингом, checkpoint и CPU-обучением.",
    "Percentage of system memory currently used by all applications.": "Доля оперативной памяти, занятая всеми приложениями.",
    "Current NVIDIA GPU compute utilization reported by the driver.": "Текущая вычислительная загрузка NVIDIA GPU по данным драйвера.",
    "Current physical GPU temperature reported by NVIDIA NVML. N/A means the sensor is unavailable.": "Текущая физическая температура GPU по данным NVIDIA NVML. Н/Д означает, что датчик недоступен.",
    "VRAM actively allocated by PyTorch tensors compared with the configured neural memory limit.": "VRAM, занятая тензорами PyTorch, относительно заданного лимита памяти нейросети.",
    "VRAM reserved by PyTorch's caching allocator for fast reuse.": "VRAM, зарезервированная кэширующим аллокатором PyTorch для быстрого повторного использования.",
    "VRAM used by the whole system compared with the GPU's physical capacity.": "VRAM, занятая всей системой, относительно физического объёма GPU.",
    "Compute device selected automatically for neural-network evaluation and evolution.": "Вычислительное устройство, автоматически выбранное для оценки нейросети и эволюции.",
    "Current lifecycle state of the training worker, including pause, evaluation, evolution, reset, and errors.": "Текущее состояние обучения: пауза, оценка, эволюция, сброс или ошибка.",
    "Controls the maximum VRAM budget used by neural tensors. AUTO derives a safe budget from detected GPU capacity and reserve settings.": "Максимальный бюджет VRAM для тензоров. АВТО рассчитывает безопасное значение по объёму GPU и резерву.",
    "Controls population size. Applying a new value starts fresh because populations of different sizes cannot share one checkpoint safely.": "Размер популяции. Новое значение начинает обучение заново, поскольку разные размеры нельзя безопасно хранить в одном checkpoint.",
    "Starting": "Запуск",
    "Pausing": "Приостановка",
    "Resuming": "Продолжение",
    "Reset requested": "Запрошен сброс",
    "Applying VRAM limit": "Применение лимита VRAM",
    "Applying agent count": "Применение числа агентов",
    "Loading checkpoint": "Загрузка checkpoint",
    "Training": "Обучение",
    "Training · new agent count": "Обучение · новое число агентов",
    "Evolving": "Эволюция",
    "Stopped": "Остановлено",
    "Complete": "Завершено",
    "Resumed": "Возобновлено",
    "Error": "Ошибка",
    "Paused": "Пауза",
    "Paused · fresh start": "Пауза · новое обучение",
    "Training · fresh start": "Обучение · новый старт",
    "Resetting for new agent count": "Сброс для нового числа агентов",
    "Resetting progress": "Сброс прогресса",
}


def normalize_language(language: str) -> str:
    return "en" if str(language).lower() == "en" else "ru"


def text(language: str, source: str, **values: object) -> str:
    template = RU_TEXT.get(source, source) if normalize_language(language) == "ru" else source
    return template.format(**values) if values else template


def language_name(language: str, value: str) -> str:
    names = {"ru": {"ru": "Русский", "en": "Английский"}, "en": {"ru": "Russian", "en": "English"}}
    return names[normalize_language(language)][normalize_language(value)]


def status_text(language: str, status: str) -> str:
    prefix = "Training · piece "
    if status.startswith(prefix):
        number = status[len(prefix):]
        return f"Обучение · фигура {number}" if normalize_language(language) == "ru" else status
    return text(language, status)


def error_text(language: str, message: str) -> str:
    if normalize_language(language) == "en":
        return message
    direct = {
        "Elite agents cannot exceed agents per generation": "Число элитных агентов не может превышать размер поколения",
        "Parent pool cannot be smaller than elite agents": "Родительский пул не может быть меньше числа элитных агентов",
        "Parent pool cannot exceed agents per generation": "Родительский пул не может превышать размер поколения",
        "Network output size must equal four rotations times board width": "Размер выхода сети должен соответствовать четырём поворотам для каждой колонки",
        "Population is unavailable": "Популяция недоступна",
    }
    if message in direct:
        return direct[message]
    if ": " in message:
        prefix, payload = message.split(": ", 1)
        localized = error_text(language, payload)
        return f"{prefix}: {localized}"
    checkpoint_prefix = "Checkpoint network architecture does not match version "
    if message.startswith(checkpoint_prefix):
        return "Архитектура сети checkpoint не соответствует версии " + message[len(checkpoint_prefix):]
    prefixes = (
        ("Missing value: ", "Отсутствует значение: "),
        ("Invalid value for ", "Некорректное значение: "),
    )
    for prefix, translated in prefixes:
        if message.startswith(prefix):
            return translated + text(language, message[len(prefix):])
    if " must be at least " in message:
        label, value = message.split(" must be at least ", 1)
        return f"{text(language, label)}: минимальное значение {value}"
    if " must not exceed " in message:
        label, value = message.split(" must not exceed ", 1)
        return f"{text(language, label)}: максимальное значение {value}"
    return message
