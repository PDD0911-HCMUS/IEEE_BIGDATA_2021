import { Component, ViewChild } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { TreeListModule } from '@progress/kendo-angular-treelist';
import { AppLanguage} from './services/app.language';
import { AppService } from './services/app.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  //styleUrls: ['./app.component.css']
})
export class AppComponent {

  @ViewChild('appMenu') appMenu;

  title = 'Sở Ngoại Vụ';
  loading = false;
  menuHidden = false;
  openMenuFlg = false;
  excLoading = true;
  user: any;
  public mbWebviewFlg = false;

  constructor(
    private language: AppLanguage,
    private router: Router,
    private activatedRoute: ActivatedRoute,
    private appService: AppService
    ) {

    localStorage.removeItem('openMenuFlg');
    if (this.appMenu) this.appMenu.openMenuFlg = '0';
    this.language.default();
  }

  public value: Date = new Date();

  switchLanguage(language: string) {
    this.language.set(language);
  }

  onReloadMenu() {
    this.menuHidden = false;
    this.appMenu.onReload();
  }

  onOpenMenu() {
    this.openMenuFlg = !this.openMenuFlg;
    var _openMenuFlg = this.openMenuFlg ? '1' : '0';
    localStorage.setItem('openMenuFlg', _openMenuFlg);
    if (this.appMenu) this.appMenu.openMenuFlg = _openMenuFlg;
  }


  showDialog = () => {
    document.getElementById("body-block").classList.add('show');
    const body = document.body;
    const scrollY = body.style.top;
    body.style.position = 'fixed';
    body.style.height = window.innerHeight.toString() + 'px';
    body.style.overflowY = 'hidden';
  };

  checkPortal() {
    if(this.router.url != "/") {
      if(this.router.url.substring(1).slice(0,6) == "portal") {
        return false;
      }
    }
    return true;
  }
  
  closeDialog = () => {
    const body = document.body;
    const scrollY = body.style.top;
    body.style.position = '';
    body.style.top = '';
    body.style.overflowY = 'auto';
    body.style.height = 'auto';
    window.scrollTo(0, parseInt(scrollY || '0') * -1);
    document.getElementById("body-block").classList.remove('show');
  }

}
