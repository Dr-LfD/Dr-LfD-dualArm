(define (domain skill_seg)
  (:requirements :strips :typing) ; Added :typing for clarity

  (:types
    robot object surface skill
  )

  (:predicates
    (LeftArm ?a - robot)
    (RightArm ?a - robot)

    ; (Region ?s - surface)
    ; (Skill ?sk - skill) <-- THIS PREDICATE IS REMOVED

    (ArmEmpty ?a - robot)
    (ArmHolding ?a - robot ?o - object)
    (On ?o - object ?s - surface)
    
    ; below generated from the language instruction
    (SkillPick ?sk - skill)  
    (SkillPlace ?sk - skill) 
    (SkillBiLR2LR ?sk - skill) 
    (SkillBiR2L ?sk - skill) 
    (SkillNudge ?sk - skill)

    (DoneSkillPick ?sk - skill ?a - robot ?o - object)
    (DoneSkillPlace ?sk - skill ?a - robot ?o - object ?s - surface)
    (DoneSkillNudge ?sk - skill ?a - robot ?o - object)
    (DoneSkillBiLR2LR ?sk - skill ?a1 - robot ?a2 - robot ?o1 - object ?o2 - object)
    (DoneSkillBiR2L ?sk - skill ?a1 - robot ?a2 - robot ?o2 - object)
  )

  (:action pick
    :parameters (?a - robot ?o - object ?s - surface ?sk - skill)
    :precondition (and
    ;   (Arm ?a)
      (ArmEmpty ?a)
      (On ?o ?s)
        (SkillPick ?sk)
    )
    :effect (and
      (ArmHolding ?a ?o)
      (DoneSkillPick ?sk ?a ?o)
      (not (ArmEmpty ?a))
      (not (On ?o ?s))
    )
  )

  (:action place
    :parameters (?a - robot ?o - object ?s - surface ?sk - skill)
    :precondition (and
    ;   (Arm ?a)
    ;   (Region ?s)
        (SkillPlace ?sk)
      (ArmHolding ?a ?o)
    )
    :effect (and
      (ArmEmpty ?a)
      (On ?o ?s)
      (DoneSkillPlace ?sk ?a ?o ?s)
      (not (ArmHolding ?a ?o))
    )
  )

  (:action nudge
    :parameters (?a - robot ?o - object ?sk - skill)
    :precondition (and
      (ArmEmpty ?a)
        (SkillNudge ?sk)
    )
    :effect (and
      (DoneSkillNudge ?sk ?a ?o)
    )
  )

    (:action BiOperationLR2LR
        :parameters (?a1 - robot ?a2 - robot ?o1 - object ?o2 - object ?sk - skill)
        :precondition (and
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
        :parameters (?a1 - robot ?a2 - robot  ?o2 - object ?sk - skill)
        :precondition (and
        (LeftArm ?a1)
        (RightArm ?a2)
        (ArmEmpty ?a1)
        (ArmHolding ?a2 ?o2)
        (SkillBiR2L ?sk)
        )
        :effect (and
        (ArmHolding ?a1 ?o2)
        (not (ArmHolding ?a2 ?o2))
        (not (ArmEmpty ?a1))
        (ArmEmpty ?a2)
        (DoneSkillBiR2L ?sk ?a1 ?a2 ?o2)
        )
    )
)


; (define (domain skill_seg)
;   (:requirements :strips :typing) ; Added :typing for clarity

;   (:types
;     robot object surface skill
;   )

;   (:predicates
;     (LeftArm ?a - robot)
;     (RightArm ?a - robot)

;     ; (Region ?s - surface)
;     ; (Skill ?sk - skill) <-- THIS PREDICATE IS REMOVED

;     (ArmEmpty ?a - robot)
;     (ArmHolding ?a - robot ?o - object)
;     (On ?o - object ?s - surface)
    
;     ; below generated from the language instruction

;     (DoneSkill ?sk - skill ?a - robot ?o - object)
;     (TgtObj ?sk - skill ?a - robot ?o - object)
;   )

;   (:action pick
;     :parameters (?a - robot ?o - object ?s - surface ?sk - skill)
;     :precondition (and
;         (TgtObj ?sk ?a ?o)
;       (ArmEmpty ?a)
;       (On ?o ?s)
;     )
;     :effect (and
;       (ArmHolding ?a ?o)
;       (DoneSkill ?sk ?a ?o)
;       (not (ArmEmpty ?a))
;       (not (On ?o ?s))
;     )
;   )

;   (:action place
;     :parameters (?a - robot ?o - object ?s - surface ?sk - skill)
;     :precondition (and
;         (TgtObj ?sk ?a ?s)
;         ; (Skill ?sk)
;       (ArmHolding ?a ?o)
;     )
;     :effect (and
;       (ArmEmpty ?a)
;       (On ?o ?s)
;       (DoneSkill ?sk ?a ?s)
;       (not (ArmHolding ?a ?o))
;     )
;   )

;   (:action nudge
;     :parameters (?a - robot ?o - object ?sk - skill)
;     :precondition (and
;       (ArmEmpty ?a)
;     )
;     :effect (and
;       (DoneSkill ?sk ?a ?o)
;     )
;   )

;     (:action BiOperationLR2LR
;         :parameters (?a1 - robot ?a2 - robot ?o1 - object ?o2 - object ?sk - skill)
;         :precondition (and
;         (LeftArm ?a1)
;         (RightArm ?a2)
;         (ArmHolding ?a1 ?o1)
;         (ArmHolding ?a2 ?o2)
;         (TgtObj ?sk ?a1 ?o1)
;         (TgtObj ?sk ?a2 ?o2)
;         )
;         :effect (and
;         (DoneSkill ?sk ?a1 ?o1)
;         (DoneSkill ?sk ?a2 ?o2)
;         )
;     )

;     (:action BiOperationR2L
;         :parameters (?a1 - robot ?a2 - robot  ?o2 - object ?sk - skill)
;         :precondition (and
;         (LeftArm ?a1)
;         (RightArm ?a2)
;         (ArmEmpty ?a1)
;         (ArmHolding ?a2 ?o2)
;         (TgtObj ?sk ?a1 ?o2)
;         (TgtObj ?sk ?a2 ?o2)
;         )
;         :effect (and
;         (ArmHolding ?a1 ?o2)
;         (not (ArmHolding ?a2 ?o2))
;         (not (ArmEmpty ?a1))
;         (ArmEmpty ?a2)
;         (DoneSkill ?sk ?a1 ?o2)
;         (DoneSkill ?sk ?a2 ?o2)
;         )
;     )
; )