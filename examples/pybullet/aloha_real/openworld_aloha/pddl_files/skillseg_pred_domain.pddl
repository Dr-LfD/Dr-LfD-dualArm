(define (domain skill_seg)
  (:requirements :strips :derived-predicates :existential-preconditions :negative-preconditions)

  (:predicates
    ;; Type predicates
    (robot ?r)
    (Movable ?o)
    (surface ?s)
    (skill ?sk)

    ;; Robot properties
    (LeftArm ?a)
    (RightArm ?a)
    (ArmHolding ?a ?o)
    (ArmEmpty ?a)  ;; <<< FIX: Declare the derived predicate here

    ;; Object properties
    (On ?o ?s)
    
    ;; Skill type predicates
    (SkillPick ?sk)  
    (SkillPlace ?sk) 
    (SkillBiLR2LR ?sk) 
    (SkillBiR2L ?sk) 
    (SkillNudge ?sk)

    ;; Goal predicates
    (DoneSkillPick ?sk ?a ?o)
    (DoneSkillPlace ?sk ?a ?o ?s)
    (DoneSkillNudge ?sk ?a ?o)
    (DoneSkillBiLR2LR ?sk ?a1 ?a2 ?o1 ?o2)
    (DoneSkillBiR2L ?sk ?a1 ?a2 ?o2)
  )

  ;; DERIVED PREDICATE: The definition of the predicate's logic.
  (:derived (ArmEmpty ?a)
    (not (exists (?o) (ArmHolding ?a ?o)))
  )


  (:action pick
    :parameters (?a ?o ?s ?sk)
    :precondition (and
      (robot ?a)
      (Movable ?o)
      (surface ?s)
      (skill ?sk)
      (ArmEmpty ?a)
      (On ?o ?s)
      (SkillPick ?sk)
    )
    :effect (and
      (ArmHolding ?a ?o)
      (DoneSkillPick ?sk ?a ?o)
      (not (On ?o ?s))
    )
  )

  (:action place
    :parameters (?a ?o ?s ?sk)
    :precondition (and
      (robot ?a)
      (Movable ?o)
      (surface ?s)
      (skill ?sk)
      (ArmHolding ?a ?o)
      (SkillPlace ?sk)
    )
    :effect (and
      (On ?o ?s)
      (DoneSkillPlace ?sk ?a ?o ?s)
      (not (ArmHolding ?a ?o))
    )
  )

  (:action nudge
    :parameters (?a ?o ?sk)
    :precondition (and
      (robot ?a)
      (Movable ?o)
      (skill ?sk)
      (ArmEmpty ?a)
      (SkillNudge ?sk)
    )
    :effect (and
      (DoneSkillNudge ?sk ?a ?o)
    )
  )
  

  (:action BiOperationLR2LR
    :parameters (?a1 ?a2 ?o1 ?o2 ?sk)
    :precondition (and
      (robot ?a1)
      (robot ?a2)
      (Movable ?o1)
      (Movable ?o2)
      (skill ?sk)
      (LeftArm ?a1)
      (RightArm ?a2)
      (ArmHolding ?a1 ?o1)
      (ArmHolding ?a2 ?o2)
      (SkillBiLR2LR ?sk)
    )
    :effect (and
      (DoneSkillBiLR2LR ?sk ?a1 ?a2 ?o1 ?o2)
    )
  )

  (:action BiOperationR2L
    :parameters (?a1 ?a2 ?o2 ?sk)
    :precondition (and
      (robot ?a1)
      (robot ?a2)
      (Movable ?o2)
      (skill ?sk)
      (LeftArm ?a1)
      (RightArm ?a2)
      (ArmEmpty ?a1)
      (ArmHolding ?a2 ?o2)
      (SkillBiR2L ?sk)
    )
    :effect (and
      (ArmHolding ?a1 ?o2)
      (not (ArmHolding ?a2 ?o2))
      (DoneSkillBiR2L ?sk ?a1 ?a2 ?o2)
    )
  )
)