"""Late event delivery cannot reopen completed recovered work."""
from assist.run_service import RunService
from manage.web import threads


def test_late_original_dispatch_after_recovery_successor_finished(tmp_path, monkeypatch):
    tid='event-thread';(tmp_path/tid).mkdir()
    service=RunService(str(tmp_path))
    monkeypatch.setattr(threads,'_runs',lambda:service)
    monkeypatch.setattr(threads,'_is_pi_thread',lambda value:False)
    monkeypatch.setattr(threads,'approval_status',lambda value:{'stage':'ready'})
    monkeypatch.setattr(threads,'_get_status',lambda value:{'stage':'ready'})
    queued=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER,'submit',lambda *args,**kwargs:queued.append(args))
    monkeypatch.setattr(threads,'_recovery_decision',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('completed work reconsidered')))
    original=service.create(tid,'general-agent','Incoming event',sender='+15555550100')
    service.claim(tid,original.id);service.transition(tid,original.id,'interrupted')
    successor=service.create(tid,'general-agent',None,work_id=original.work_id,sender=original.sender,resume=True)
    service.claim(tid,successor.id);service.transition(tid,successor.id,'success')
    before=service.list(tid)
    threads._execute_run_active(original.id,tid)
    assert service.list(tid)==before
    assert queued==[]
