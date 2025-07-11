# Kategoryzator Transakcji (Web)

Prosta aplikacja do kategoryzowania transakcji bankowych w przeglądarce.

## Jak to działa?

1. Wczytujesz plik CSV z kolumnami `Date`, `Description`, `Amount`.
2. Aplikacja automatycznie przypisuje kategorie na podstawie poprzednich wyborów.
3. Możesz ręcznie poprawić każdą transakcję w interaktywnym edytorze.
4. Pobierasz gotowy plik CSV z dodanymi kolumnami `category` i `subcategory`.

## Wdrożenie na Streamlit Community Cloud

1. Utwórz nowe repozytorium na GitHub o nazwie `KategoryzatorTransakcjiWeb`.
2. Wgraj do niego pliki `app.py`, `requirements.txt`, `README.md`.
3. Przejdź na https://share.streamlit.io i zaloguj się przez GitHub.
4. Kliknij **New app**, wybierz swoje repozytorium, branch `main` i plik `app.py`.
5. Kliknij **Deploy** – po chwili aplikacja będzie działać w Twojej przeglądarce.

Gotowe! Każdy może teraz korzystać z Twojej aplikacji bez instalacji czegokolwiek lokalnie. Jeśli coś nie działa, daj znać!```

---

**Teraz w Twoim folderze `KategoryzatorTransakcjiWeb` powinny być trzy pliki**:

- `app.py`  
- `requirements.txt`  
- `README.md`  

Dalej postępujesz zgodnie z instrukcją w README: zakładasz GitHub, uploadujesz pliki, a potem deploy na Streamlit Community Cloud. Powodzenia!
