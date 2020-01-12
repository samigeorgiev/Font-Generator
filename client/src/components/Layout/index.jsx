import React, {Component} from 'react';

import {NavLink} from 'react-router-dom';

import DrawerToggle from 'components/navigation/SideDrawer/DrawerToggle';
import Logo from 'components/Logo';
import Toolbar from 'components/navigation/Toolbar';

import styles from './index.module.css';

class Layout extends Component {
    state = {
        isSideDrawerOpen: false
    }

    render() {
        const links = [
            { to: '/', value: 'Home' },
            { to: '/generator', value: 'New font' }
        ];

        if (this.props.isAuth) {
            links.push({ to: '/saved', value: 'Saved fonts' });
        } else {
            links.push({ to: '/auth', value: 'Authenticate' });
        }

        return (
            <div className={styles.Layout}>
                <header>
                    <DrawerToggle />
                    <div className={styles.FGLogo}>
                        <NavLink to="/" exact><Logo src="FGLogo"/></NavLink>
                    </div>
                    <Toolbar links={links} />
                    <div className={styles.GitHubLogo}>
                        <NavLink to="/" exact><Logo src="GitHubLogo"/></NavLink>
                    </div>
                </header>
                {this.props.children}
                <footer>
                    <p>&copy; 2019 - {new Date().getFullYear()} All rights reserved | Sami</p>
                </footer>
            </div>
        );
    }
}

export default Layout;