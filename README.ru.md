# xrd-analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23892876.svg)](https://doi.org/10.5281/zenodo.23892876)

[English version (README.md)](README.md)

**Автономный кристаллографический анализ порошковых дифрактограмм и стадийное уточнение методами Ритвельда и Паули**

`xrd-analysis` — это специализированный навык ИИ-агента (`SKILL.md`) и программный кристаллографический комплекс на языке Python, работающий поверх вычислительного ядра [GSAS-II](https://github.com/AdvancedPhotonSource/GSAS-II) (`GSASIIscriptable`), библиотек `pymatgen`, `gemmi` и `spglib`.

- **Основной репозиторий (GitFlic):** [https://gitflic.ru/project/i-o-yurev/xrd-analysis](https://gitflic.ru/project/i-o-yurev/xrd-analysis)
- **Зеркало репозитория (GitHub):** [https://github.com/ioyurev/xrd-analysis](https://github.com/ioyurev/xrd-analysis)

## Основные возможности

- **Первичная диагностика экспериментального профиля (`scripts/xrd_inspect.py`):** оценка углового диапазона $2\theta$, шага сканирования, отношения сигнал/шум (SNR), полуширины рефлексов (FWHM) на малых и больших углах, степени кристалличности и наличия гало аморфных фаз.
- **Поиск структур и генерация изоструктурных аналогов (`scripts/cif_retriever.py`):** параметрический поиск в открытой кристаллографической базе COD и базе Materials Project; построение изоструктурных аналогов путём масштабирования по ионным радиусам Шеннона или экспериментальным межплоскостным расстояниям $d_{hkl}$ с точным сохранением пространственной группы через `gemmi`.
- **Аналитический фазовый поиск Search-Match (`scripts/search_match.py`):** многокомпонентный скоринг кандидатов против сырых дифрактограмм и разностных кривых Ритвельда с использованием локальных пулов CIF, локальных SQLite-архивов COD (`COD_HSRDB_PATH`) и пользовательских установок базы ICDD PDF-2 (`PDF2_DATA_DIR`).
- **Рабочее пространство образца (`scripts/xrd_sample.py`):** организация работы по принципу «одна папка — один образец» (`data/`, `params.json`, `projects/`, `artifacts/`) с 12-стадийным протоколом уточнения Ритвельда (Stage-Gate), полнопрофильным бесструктурным разложением методом Паули (`--pawley`), декларативным управлением параметрами через `params.json`, автоматическим откатом при расходимости, клинической диагностикой разностной кривой (включая распознавание аппаратных $\text{Cu}\,K\beta$-сателлитов) и кристаллохимическим контролем по методу валентностей связей (BVS, индекс глобальной нестабильности GII).

## Установка

Требуется Python 3.11 и пакетный менеджер [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python scripts/install_gsas2.py
```

### Настройка переменных окружения (`.env`)

Подключение внешних и локальных баз данных задаётся через переменные окружения или файл `.env` в рабочей директории (проприетарные базы данных в состав репозитория не входят):

```ini
# Опционально: путь к локальному SQLite-архиву структур COD (CC0)
COD_HSRDB_PATH=/path/to/COD24_HS4x.hsrdb

# Опционально: путь к локальной лицензионной установке ICDD PDF-2 пользователя
PDF2_DATA_DIR=/path/to/PDF2
```

## Быстрый старт

Подробный операционный регламент и справочник команд приведены в [`SKILL.md`](SKILL.md):

```bash
# 1. Инициализация рабочего пространства образца из файла измерений
uv run python scripts/xrd_sample.py --dir ./my_sample init --data data/scan.xy --bkg-terms 6

# 2. Добавление кристаллической фазы (из CIF, локального архива COD или карточки PDF-2)
uv run python scripts/xrd_sample.py --dir ./my_sample add-phase --cif cifs/CeO2.cif --name CeO2

# 3. Запуск поэтапного уточнения по протоколу Stage-Gate
uv run python scripts/xrd_sample.py --dir ./my_sample step --sequence "1,2,3,5,7,12" --cycles 5

# 4. Проверка статуса, диагностика остатка и экспорт результатов (CIF, CSV, график 600 DPI, отчёт IUCr)
uv run python scripts/xrd_sample.py --dir ./my_sample status
uv run python scripts/xrd_sample.py --dir ./my_sample diagnose --out
uv run python scripts/xrd_sample.py --dir ./my_sample export
```

## Лицензия и цитирование

Пакет распространяется на условиях лицензии [BSD 3-Clause License](LICENSE) ([перевод на русский язык](LICENSE.ru)). При использовании комплекса `xrd-analysis` в научных работах, пожалуйста, цитируйте его согласно метаданным в [`CITATION.cff`](CITATION.cff) совместно с базовыми публикациями вычислительного ядра и открытых кристаллографических баз данных:

- **xrd-analysis:** Yurev, I. O. (2026). *xrd-analysis: Autonomous Crystallographic Analysis and Stage-Gate Rietveld/Pawley Refinement Skill* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23892876
- **GSAS-II:** Toby, B. H. & Von Dreele, R. B. (2013). *GSAS-II: the genesis of a modern open-source all purpose crystallography software package*. Journal of Applied Crystallography, 46, 544–549. https://doi.org/10.1107/S0021889813003531
- **Crystallography Open Database (COD):**
  - Gražulis, S., Daškevič, A., Merkys, A., Chateigner, D., Lutterotti, L., Quirós, M., Serebryanaya, N. R., Moeck, P., Downs, R. T. & Le Bail, A. (2012). *Crystallography Open Database (COD): an open-access collection of crystal structures and platform for world-wide collaboration*. Nucleic Acids Research, 40, D420–D427. https://doi.org/10.1093/nar/gkr900
  - Gražulis, S., Chateigner, D., Downs, R. T., Yokochi, A. F. T., Quirós, M., Lutterotti, L., Manakova, E., Butkus, J., Moeck, P. & Le Bail, A. (2009). *Crystallography Open Database — an open-access collection of crystal structures*. Journal of Applied Crystallography, 42, 726–729. https://doi.org/10.1107/S0021889809016690
- **AMCSD (при использовании минералогических выборок COD):** Downs, R. T. & Hall-Wallace, M. (2003). *The American Mineralogist Crystal Structure Database*. American Mineralogist, 88, 247–250.

## Правовое уведомление

PDF-2 и ICDD являются зарегистрированными товарными знаками International Centre for Diffraction Data. Настоящее программное обеспечение является независимым инструментом функциональной совместимости, не содержит проприетарных баз данных и не связано с ICDD.
