(define (problem skill_seg_problem)
  (:domain skill_seg)

  (:objects
    r1 r2 obj1 obj2 drawer loc1 loc2 skPick skPlace skPush skInsertion skHandOff
  )

  (:init
    ;; Type definitions as predicates
    (robot r1)
    (robot r2)
    (Movable obj1)
    (Movable obj2)
    (Movable drawer)
    (surface loc1)
    (surface loc2)
    (skill skPick)
    (skill skPlace)
    (skill skPush)
    (skill skInsertion)
    (skill skHandOff)

    ;; Object properties
    (LeftArm r1)
    (RightArm r2)
    (SkillPick skPick)  
    (SkillPlace skPlace) 
    (SkillBiLR2LR skInsertion) 
    (SkillBiR2L skHandOff) 
    (SkillNudge skPush)

    ;; Initial state of the world
    ;(ArmEmpty r1)
    (On obj1 loc1)
    ;(ArmEmpty r2)
    (On obj2 loc1)
  )

  (:goal
    (and
    ;   ;; push the drawer and pick the obj2 
    ;   (DoneSkillNudge skPush r1 drawer)
    ;   (DoneSkillPick skPick r2 obj2)

    ; ; ; LR2LR
    ;   (DoneSkillBiLR2LR skInsertion r1 r2 obj1 obj2)

      ; pick place x2
      (DoneSkillPick skPick r1 obj1)
      (DoneSkillPlace skPlace r1 obj1 loc2)
      (DoneSkillPick skPick r1 obj2)
      (DoneSkillPlace skPlace r1 obj2 loc2)
    )
  )
)
