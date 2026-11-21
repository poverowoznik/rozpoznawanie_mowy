# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 6 — Projekt generacyjny, część 1: przygotowanie danych dla Char-RNN
**Autor / grupa:** Jakub Powroźnik Gr Lab 5.03
**Data:** 21.11.2026

## 1. Cel ćwiczenia

Celem było przygotowanie polskiego korpusu tekstowego do zadania przewidywania następnego znaku (Char-RNN), które zostanie użyte na zajęciach 7 (trening) i 8 (generowanie). Zakres pracy: wybór i oczyszczenie korpusu, podział na zbiory train/val/test **przed** utworzeniem okien, słownik znaków zbudowany na treningu, obsługa znaków spoza słownika, `Dataset` i `DataLoader` dla każdego zbioru, kontrola batchy oraz zbadanie wpływu długości okna `SEKW` na liczbę par.

## 2. Metoda i implementacja

**Narzędzia:** Python 3.12, PyTorch (`Dataset`, `DataLoader`), NumPy, pandas, matplotlib, `unicodedata`, `re`.

### 2.1. Korpus

**Źródło:** George Orwell, *Rok 1984*, tłum. Julia Fiedorczuk, plik TXT z serwisu Wolne Lektury (`https://wolnelektury.pl/media/book/txt/orwell-rok-1984.txt`).

**Warunki użycia:** według stopki pliku utwór jest udostępniony na **Licencji Wolnej Sztuki 1.3** (https://artlibre.org/licence/lal/pl/), wydawca: Fundacja Wolne Lektury. Zasady serwisu dopuszczają pobieranie plików i używanie ich w pracach zaliczeniowych pod warunkiem zachowania atrybucji. *Książka pochodzi z serwisu Wolne Lektury.*

**Uzasadnienie wyboru:** powieść jest dostatecznie długa (ok. 560 tys. znaków), żeby po podziale 80/10/10 zbiory val i test miały po ponad 55 tys. znaków, czyli dużo więcej niż największe badane okno (160). Tekst jest spójny stylistycznie i napisany współczesną polszczyzną, z polskimi znakami diakrytycznymi.

**Charakterystyka:** powieść dystopijna, proza narracyjna z dialogami (długa półpauza `—` na początku wypowiedzi), z krótkimi akapitami i dłuższym esejem w Aneksie (Zasady nowomowy), który jest nieco innym stylem niż reszta.

### 2.2. Przygotowanie tekstu

Pobrany plik miał 567 766 znaków. Przekształcenia, w kolejności:

| Krok | Co zrobiono | Długość po kroku |
|---|---|---|
| 1 | Obcięcie przedmowy tłumaczki, nagłówka i ISBN (od zdania „Był jasny, zimny kwietniowy dzień"); obcięcie stopki Wolnych Lektur (od `-----\r\nTa lektura`) i `rstrip()` | 562 741 |
| 2 | Normalizacja Unicode **NFC** (rozłożone litery z 3 znakami łączącymi → gotowe litery) | 562 706 |
| 3 | `\r\n` → `\n` (2898 par, zawsze występowały razem) | 559 808 |
| 4 | Usunięcie `*` (552 znaczniki kursywy z Wolnych Lektur) | 559 256 |
| 5 | Zwinięcie 3+ znaków nowej linii w dwa (74 miejsca) | 559 132 |
| 6 | Poprawa 4 literówek `–-` / `-–` na początku dialogów → `—` (zostawiona 1 półpauza w zakresie `1920–1940`) | **559 128** |

Kontrola początku, środka i końca tekstu pokazała, że w pliku poza powieścią znajdowały się: przedmowa tłumaczki, dane wydawnicze i długa lista darczyńców (nazwiska) w stopce. Takie fragmenty zostałyby przez model zapamiętane i generowane jako listy nazwisk, więc zostały usunięte.

Fragment po oczyszczeniu:

```
Był jasny, zimny kwietniowy dzień, zegary wybijały trzynastą. Winston Smith, przyciskając
brodę do piersi dla osłony przed zacinającym wiatrem, wślizgnął się najszybciej, jak
potrafił, przez szklane drzwi Rezydencji Zwycięstwa; ...
```

Decyzje i ich powody:
- Aneks (*Zasady nowomowy*) **zostawiony**, bo jest częścią książki.
- Wielkość liter **bez zmian**, bo wielkie litery i interpunkcja niosą informację o stylu.
- `\r\n` → `\n`: jeden znak nowej linii zamiast dwóch, mniejszy słownik.

### 2.3. Podział danych

Podział jest wykonany **na tekście**, przed utworzeniem okien (aby uniknąć przecieku danych, bo sąsiednie okna w większości się pokrywają). Jest to jeden długi tekst, więc wydzielono trzy rozłączne, ciągłe fragmenty w proporcji **80% / 10% / 10%**. Granice przesunięto do najbliższego końca akapitu (`\n\n`), żeby nie rozcinać zdań.

Kod: notatnik, sekcja „4. Podział na train / val / test" (funkcja `koniec_akapitu` wyznacza granice na końcu akapitu).

Proporcje: trening ma największy udział, bo to na nim model uczy się wag. Val i test po ok. 56 tys. znaków dają wystarczająco dużo par dla każdej badanej wartości `SEKW`.

### 2.4. Słownik i kodowanie

Słownik budowany **wyłącznie na zbiorze treningowym** i użyty we wszystkich zbiorach. Znaki spoza słownika dostają specjalne ID `<unk>` (ostatnie ID).

Kod: notatnik, sekcje „4. Podział…" (budowa `stoi` i `itos` z `train`) oraz „5. Słownik (tylko z treningu) i obsługa unk" (funkcje `enc` i `dec`, `UNK_ID`).

### 2.5. Dataset i DataLoader

Kod: notatnik, sekcja „6. Dataset, DataLoader i kontrola batchy" (klasa `DatasetZnakowy`: wejście `x = dane[i : i + SEKW]`, cel `y = dane[i + 1 : i + SEKW + 1]`, czyli okno przesunięte o 1 znak).

**Konfiguracja:** `SEKW = 40`, `batch_size = 32`, krok okna = 1. Train: `shuffle=True`, `drop_last=True`. Val i test: `shuffle=False`, `drop_last=False` (ocena obejmuje wszystkie przykłady). Okna powstają wewnątrz każdego zbioru osobno.

## 3. Wyniki

### 3.1. Podział i liczebności (SEKW = 40, batch = 32)

| Zbiór | Znaków | Udział | Różnych znaków | Par | Batchy | Dokumentów |
|---|---|---|---|---|---|---|
| train | 447 326 | 80,0% | 94 | 447 286 | 13 977 | 1 (fragment) |
| val | 56 681 | 10,1% | 76 | 56 641 | 1 771 | 1 (fragment) |
| test | 55 121 | 9,9% | 88 | 55 081 | 1 722 | 1 (fragment) |

Korpus to jedna książka, więc zbiory są ciągłymi fragmentami tego samego dokumentu.

### 3.2. Kontrola danych

**Rozmiar słownika:** 94 znaki z treningu + `<unk>` = **95**.

**Znaki spoza słownika:** val: `Ę` (1×), test: `=` (1×) i `+` (1×). Razem 3 wystąpienia obsłużone przez `<unk>`; w treningu 0.

**Test kodowania i dekodowania:** `dec(enc(tekst))` odtwarza tekst we wszystkich trzech zbiorach (z `<unk>` w miejscach znaków obcych): train, val, test — **True**.

**Kształt i typ batcha:** w każdym zbiorze `X` i `Y` mają kształt `(32, 40)`, typ `torch.int64`, ID w granicach słownika (0–94).

**Sprawdzone pary** (indeks 1000 w każdym zbiorze, porównane z tekstem źródłowym, w tym ostatni znak celu):

| Zbiór | Wejście | Cel |
|---|---|---|
| train | `'u dokuczało mu owrzodzenie żylne nad pra'` | `' dokuczało mu owrzodzenie żylne nad praw'` |
| val | `'\n\n— Czasem tak, Winstonie. Ale czasem dw'` | `'\n— Czasem tak, Winstonie. Ale czasem dwa'` |
| test | `'otem ciągnął łagodniejszym tonem:\n\n— Rob'` | `'tem ciągnął łagodniejszym tonem:\n\n— Robi'` |

Dla każdej pary spełniony jest warunek `x[1:] == y[:-1]`, a oba okna są zgodne z fragmentem tekstu od indeksu 1000.

### 3.3. Wpływ długości okna SEKW (krok = 1, ten sam tekst i podział)

| SEKW | Pary train | Pary val | Pary test | Batchy train | Batchy val | Batchy test |
|---|---|---|---|---|---|---|
| 20 | 447 306 | 56 661 | 55 101 | 13 978 | 1 771 | 1 722 |
| 40 | 447 286 | 56 641 | 55 081 | 13 977 | 1 771 | 1 722 |
| 80 | 447 246 | 56 601 | 55 041 | 13 976 | 1 769 | 1 721 |
| 160 | 447 166 | 56 521 | 54 961 | 13 973 | 1 767 | 1 718 |

![Liczba par w zależności od SEKW](img/pary_sekw.png)

![Ilość danych wejściowych na epokę](img/znaki_na_epoke.png)

### 3.4. Najczęstsze znaki w treningu

![20 najczęstszych znaków](img/znaki_top20.png)

### 3.5. Słowa

W treningu jest 63 979 słów, w tym 16 645 unikalnych. Najczęstsze: `się` (1903), `w` (1733), `nie` (1405), `i` (1258), `na` (1061), `z` (1025), `to` (778), `że` (770), `do` (733), `o` (527); z powieści `winston` (394). Średnia długość słowa: **5,73** znaku.

![Najczęstsze słowa i rozkład długości słów](img/slowa.png)

## 4. Analiza i interpretacja

**Liczba par a SEKW.** Przy kroku 1 liczba par wynosi `N − SEKW`, co potwierdza tabela (np. train: 447 326 − 160 = 447 166). Przejście z `SEKW` = 20 na 160 odbiera tylko ok. 0,03% par w treningu i ok. 0,3% w val i teście, więc liczba par praktycznie nie zależy od długości okna. Zmienia się natomiast koszt: każda para ma `SEKW` znaków, więc ilość danych wejściowych na epokę rośnie liniowo (dla `SEKW` = 160 jest 8 razy większa niż dla 20, ok. 71,5 mln wobec ok. 9 mln znaków w treningu). Dłuższe okno daje modelowi więcej kontekstu, ale sąsiednie okna w większości się powtarzają, więc epoka trwa dłużej bez nowej informacji.

**Znaki.** Najczęstszym znakiem jest spacja (64 056 wystąpień), dalej `i`, `a`, `e`, `o`, `n`, `z`. Rozkład jest mocno skośny: kilka znaków dominuje, a ogon rzadkich znaków (`â`, `À`, `=`, `+`) ma po 1-2 wystąpienia. Stąd znaki, które trafiły do val/test, a nie ma ich w treningu, i konieczność obsługi `<unk>`. Zbiory val i test mają mniej różnych znaków (76 i 88) niż trening (94), bo są krótsze.

**Słowa.** Tylko ok. 26% słów jest unikalnych, co wynika z polskiej odmiany. Najczęstsze słowa to krótkie słowa funkcyjne. Rozkład długości ma maksimum przy 5 literach i prawy ogon. Ze średniej długości słowa ze spacją (ok. 6,7 znaku) wynika, że okno 40 znaków obejmuje zaledwie ok. 6 słów, a 160 znaków ok. 24. Model znakowy z krótkim oknem widzi więc niewielki kontekst.

**Ograniczenia.**
- Jedna książka, jeden autor i jeden tłumacz: zbiory są ciągłymi fragmentami tego samego tekstu. Val i test pochodzą z końca powieści (m.in. zakończenie i Aneks o nowomowie), więc mogą różnić się stylem od treningu.
- Rzadkie znaki są zastępowane przez `<unk>`, dlatego dokładne odtworzenie tekstu jest możliwe tylko na fragmencie pokrytym słownikiem.
- Okna z krokiem 1 silnie się pokrywają, dlatego podział wykonano na tekście, a nie na gotowych oknach.

## 5. Wnioski

1. Potok przygotowania danych działa poprawnie: kodowanie jest odwracalne, cel jest przesunięty o jeden znak, a kształty, typy i zakresy ID zgadzają się z konfiguracją we wszystkich trzech zbiorach.
2. Czyszczenie było konieczne: przedmowa, dane wydawnicze, stopka z listą nazwisk, znaczniki `*`, `\r` i znaki łączące (po NFC) zmniejszyły słownik ze 102 do 97 znaków i usunęły treści niebędące prozą.
3. Podział na tekście przed oknami chroni przed przeciekiem danych. Słownik z treningu plus `<unk>` (3 obsłużone wystąpienia) pozwala użyć tego samego mapowania we val i teście.
4. Długość okna prawie nie zmienia liczby par, ale liniowo zwiększa koszt epoki. Na zajęciach 7 trzeba zmierzyć czas epoki i wybrać `SEKW` jako kompromis między kontekstem a czasem.
5. Kierunki poprawy: większy lub bardziej różnorodny korpus (kilka książek, podział po dokumentach), krok okna większy niż 1, żeby ograniczyć powtórzenia, oraz sprawdzenie, czy Aneks wpływa na wyniki.

---

*Źródło tekstu: Wolne Lektury, G. Orwell, „Rok 1984", tłum. J. Fiedorczuk, Licencja Wolnej Sztuki 1.3. Książka pochodzi z serwisu Wolne Lektury.*
