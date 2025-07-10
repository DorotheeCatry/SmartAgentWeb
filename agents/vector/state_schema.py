from typing import TypedDict, Optional, List, Dict

class GraphState(TypedDict, total=False):
    """Schéma d'état pour le graphe RH"""
    query: str
    data_analytics: Optional[Dict]
    recruiter: Optional[List[Dict]]
    rh: Optional[str]
    talent: Optional[Dict]
    onboarding: Optional[Dict]
    payroll: Optional[Dict]
    critique: Optional[str]
    validation: Optional[Dict]
    final_answer: Optional[Dict]