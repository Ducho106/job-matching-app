import numpy as np
from main import cosine_similarity, durchschnittliche_aehnlichkeit
from unittest.mock import patch, MagicMock
from main import hole_chat_historie


def test_cosine_similarity_identische_vektoren():
    vektor = np.array([1, 2, 3])
    ergebnis = cosine_similarity(vektor, vektor)
    assert ergebnis == 1.0

def test_cosine_similarity_unterschiedliche_vektoren():
    vektor_a = np.array([1,0])
    vektor_b = np.array([0,1])
    ergebnis = cosine_similarity(vektor_a, vektor_b)
    assert ergebnis == 0

def test_durchschnittliche_aehnlichkeit_leere_liste():
    ergebnis = durchschnittliche_aehnlichkeit("Maschinenbau", [])
    assert ergebnis == 0



def test_hole_chat_historie():
    fake_nachricht = MagicMock()
    fake_nachricht.rolle = "user"
    fake_nachricht.inhalt = "Hallo"
    
    with patch("main.session") as mock_session:
        mock_session.query.return_value.filter.return_value.all.return_value = [fake_nachricht]
        
        ergebnis = hole_chat_historie(1)
        
        assert ergebnis == [{"role": "user", "parts": [{"text": "Hallo"}]}]